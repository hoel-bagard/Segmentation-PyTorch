## Original data format
The original data format contains folders with the following structure:

```
image_name
  ├── pic 
  │   ├── 0101.bin    # Disparity map for the first tile
  │   ├── 0101.bmp    # Grayscale image for the first tile
  │   ├── ...         # Etc
  │   ├── 1007.bmp
  │   └── 1007.bmp
  └── annotation.csv
```
  
The tile names are 2 zero-padded ints, indicating the x and y coordinates of the tile.\
The csv contains 3 columns: tile name, tile class, tile danger level.


## Conversion/Preprocessing
To have data in a more conventionnal format, the tiles first need to be stitched together (there is some overlapp between them).\
Then we generate segmentation masks. The masks are pretty small since we only have the class of the tiles, not for each pixel.\
Finally split the dataset into Train and Validation.

#### Stitching
Stitch together the tiles with:

```
python utils/stitch.py <path to the dataset> <path to where the stiched images will be saved>
```

Example:

```
python utils/stitch.py ../data/20220128 ../data/preprocessed
```

#### Generate the masks
There are two segmentations masks per image for this project:
- An RGB mask used for classification.
- A grayscale one used for danger level prediction.
The masks are 8 by 5 pixels since this is the label precision we got.\
TODO: It might be possible to get a 16x10 mask by taking into account the overlapp. For each "pixel" get the value from each tile that contains it, then take the class/level that appears the most.

Generate segmentation masks with (make sure the classes and max danger level are set in the data config file):

```
python -m utils.create_segmentation_masks <path to the dataset> <path to where the stiched masks will be saved>
```

Example:

```
python -m utils.create_segmentation_masks ../data/20220128/ ../data/preprocessed
```

Note: You can add the `-f` option to generate full size masks. This is not necessary, but helps a lot for visualization / to check for bugs (especially since the danger mask's values go from 0 to 5, so basically everything is black).\
Note: For the danger level mask, the values in the original dataset are 1, 3 and 5 but in the generated mask they are 0, 1 and 2.

#### Output example
| Image | Disparity | Classification Mask | Danger Mask |
|    :---:      |    :---:      |    :---:      |    :---:      |
| ![image1](https://drive.google.com/uc?export=view&id=1QZvQHUcH0LZzFIFqC3YbPnhhps_xE1S5) | ![disparity1](https://drive.google.com/uc?export=view&id=1vaZ2DHmmA5r3WqF0S_PAPWtfgioAMBzS) | ![cls_mask1](https://drive.google.com/uc?export=view&id=1gpjI513nsZdhxMxviQ54bBvAZgf49vjt) | ![danger_mask1](https://drive.google.com/uc?export=view&id=1legzSvU93XZcK2_lQu3gHYZtW6mw5Xry) |
| ![image2](https://drive.google.com/uc?export=view&id=1Q79yZEc2Wfv9MRZ2hAY45DXAQRUQbac0) | ![disparity2](https://drive.google.com/uc?export=view&id=10T0KrNNOmjeMEoT184RtR3mBK_BtUs3N) | ![cls_mask2](https://drive.google.com/uc?export=view&id=17o2IjsilXmuCKCJRic4c6p9Ub7H96GmD) | ![danger_mask2](https://drive.google.com/uc?export=view&id=1CnvohZWSh8upF0pea-9SMFkaEtUCO8Vw) |


#### Split the data
You need to split the data between two folders: "Train" and "Validation" (the names are hard coded). You can either do it by hand, or modify the `utils/split_train_val.py` script to work on this dataset.\
I did it manually for simplicity.

## Loading
The data is loaded in `danger_p_loader.py`.\
For each sample, the grayscale image and the disparity map are stacked to get an "image" with 2 channels.\
For the segmentation maps, they are first transfmorded into one hot masks and then stacked (The one hot danger mask is padded to have the same shapped as the number of classes, which means that there must be more classes than danger levels). This leads to a shape of `(batch size, height, width, nb classes, 2)` which is not ideal as images usually do not have 4 dimensions.

TODO: Having a shape `(batch size, height, width, nb classes + max danger level)` would most likely have been better. Would have made handling the data augmentation, loss, etc... must easier. The only part that would have been different from a normal segmentation would have been the argmax part (plus drawing/metrics).
