The dataset.py script contains a small utility class object, MaskDataset(), for cleaning segmentation masks. Specify your `root_dir` argument and structure the directory as follows:

```text
root_dir/
├── resolution.xlsx          # Excel file with voxel resolution for each image
├── model_prediction/        # Unprocessed prediction masks
│   ├── group1_img01_segmentation.tif
│   ├── group1_img02_segmentation.tif
│   └── group2_img01_segmentation.tif
└── processed_prediction/    # Output folder for processed masks
```

For `resolution.xlsx`, specify each image's X, Y, and Z resolutions in um^3/pixel:

```text
┌────────────────────┬──────┬──────┬──────┐
│ Image              │ resz │ resy │ resx │
├────────────────────┼──────┼──────┼──────┤
│ group1_img01.tif   │ 0.8  │ 0.2  │ 0.2  │
│ group1_img02.tif   │ 0.8  │ 0.2  │ 0.2  │
│ group2_img01.tif   │ 1.0  │ 0.5  │ 0.5  │
└────────────────────┴──────┴──────┴──────┘
```

Finally, use the `save_to_disk` method to save all processed masks to the `processed_prediction/` directory.
