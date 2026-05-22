import numpy as np
import pandas as pd
from skimage.measure import label
from skimage.morphology import binary_closing, ball, remove_small_objects, remove_small_holes
import tifffile
import os

class MaskDataset():
    """
    Dataset class for processing predicted segmentation masks.
    """
    def __init__(self, root_dir, filter_obj_size=100, filter_hole_size=10):
        """
        Args:
            - root_dir (str): Root directory containing the resolution table and predicted masks.
            - filter_obj_size (int): Size threshold for removing small objects (in um^3).
            - filter_hole_size (int): Size threshold for filling small holes (in um^3).
        """
        self.root_dir = root_dir
        self.filter_obj_size = filter_obj_size
        self.filter_hole_size = filter_hole_size

        self.resolution_table = pd.read_excel(os.path.join(root_dir, 'resolution.xlsx'), sheet_name='Sheet1') # Resolution in um
        self.resolution_table['group'] = self.resolution_table['Image'].apply(lambda x: x.split('_')[0])
        self.resolution_table['voxel_size'] = self.resolution_table.apply(lambda x: x['resz'] * x['resy'] * x['resx'], axis=1)
        self.resolution_table['voxel_cutoff_obj_size'] = self.resolution_table['voxel_size'].apply(lambda x: x * filter_obj_size)
        self.resolution_table['voxel_cutoff_hole_size'] = self.resolution_table['voxel_size'].apply(lambda x: x * filter_hole_size)

        self.resolution_table.index = self.resolution_table['Image'].apply(lambda x: x.split('.')[0])
        self.imgs = self.resolution_table.index.tolist()
        self.groups = [img.split('_')[0] for img in self.imgs]
        self.labels = {}

        for img_name in self.imgs:
            voxel_cutoff_obj_size = self.resolution_table.loc[img_name, 'voxel_cutoff_obj_size']
            voxel_cutoff_hole_size = self.resolution_table.loc[img_name, 'voxel_cutoff_hole_size']
            img_path = os.path.join(root_dir, 'model_prediction', img_name + '_segmentation.tif')
            self.labels[img_name] = self.processing_pipeline(img_path, voxel_cutoff_obj_size, voxel_cutoff_hole_size)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.labels[self.imgs[key]]
        if isinstance(key, str):
            return self.labels[key]
        else:
            raise KeyError("Key must be either an integer index or a string image name.")

    def read_tiff_as_numpy(self, img_path):
        # Images are (z, y, x) shaped
        try:
            image_array = tifffile.imread(img_path)
            return image_array
        except Exception as e:
            print(f"Error reading TIFF file: {e}")
            return None
        
    def fill_cavities(self, image_array):
        # Note: Only works pre-labeling
        processed_mask = binary_closing(image_array, footprint=ball(1))
        return processed_mask

    def obtain_labels(self, image_array, connectivity=1):
        labels = label(image_array, connectivity=connectivity)
        return labels

    def destroy_small_objects(self, labels, connectivity=1, max_size=64):
        processed_labels = remove_small_objects(labels, connectivity=connectivity, min_size=max_size)
        return processed_labels

    def fill_small_holes(self, labels, connectivity=1, max_size=64):
        processed_labels = remove_small_holes(labels, connectivity=connectivity, area_threshold=max_size)
        return processed_labels
    
    def processing_pipeline(self, img_path, voxel_cutoff_obj_size, voxel_cutoff_hole_size):
        image_array = self.read_tiff_as_numpy(img_path)
        if image_array is None:
            return None
        image_array = self.fill_cavities(image_array)
        
        labels = self.obtain_labels(image_array)
        labels = self.destroy_small_objects(labels, connectivity=1, max_size=voxel_cutoff_obj_size)
        image_array = self.fill_small_holes(labels, connectivity=1, max_size=voxel_cutoff_hole_size)
        labels = self.obtain_labels(image_array)
        return labels
    
    def quantify_volume(self):
        volumes = []
        voxel_count = []
        groups = [g.split('_')[0] for g in list(self.labels.keys())]
        for img in self.labels.keys():
            voxel_res = self.resolution_table.loc[img]['voxel_size']
            current_voxel_count = np.sum(self.labels[img] > 0) 
            voxel_count.append(current_voxel_count)
            volumes.append(current_voxel_count * voxel_res)
        results_dict = {'Image': list(self.labels.keys()),
                        'Group': groups,
                        'voxel_count': voxel_count,
                        'volume_um3': volumes}
        return pd.DataFrame(results_dict)

    def save_to_disk(self):
        for img in self.labels.keys():
            if self.labels[img] is None:
                continue
            arr = self.labels[img].astype(bool).astype('uint8')
            tifffile.imwrite(os.path.join(self.root_dir, 'processed_prediction', img + '.tif'), arr)