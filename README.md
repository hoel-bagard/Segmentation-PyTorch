# Segmentation-PyTorch

Things I want to change:
- The standardization values.
- The way the disparity maps are assembled.
- Clean the disparity maps as a preprocessing step.
- The initialisation of the network. (currently there isn't any)
- Why is the model returning the latent ?
- The model itself makes little sense imho.
- The loss. (although I haven't much experience with losses and tiny masks)


## Installation

### Dependencies
torch\
torchsummary\
tensorboard\
opencv-python

### Clone the repository
```
git clone git@github.com:hoel-bagard/Segmentation-PyTorch.git --recurse-submodules
```

## Project Goal
TODO

## Config files
In the config folder of this repo you will find two config template files. You need to copy them and remove the "_template" part.

### DataConfig
Contains most of the parameters regarding the data. Most of the values in the template can be kept as they are. But some things need to be set:
- The class names and colors for the dataset (if a class is not included there, then it will be converted to "その他" when generating the masks).
- The 3 paths usually need to be modified for each training (`DATA_PATH`, `CHECKPOINT_DIR` & `TB_DIR`). 

### ModelConfig
Contains the parameters that influence training. The default values should work fine, but you can try to tweak them to get better results.\
For the `MAX_EPOCHS` value, usually around 400 or 600 epochs is enough, you will need to train at least once to get an idea for your particular dataset.

### Get some data and format it:

The data for this project is on the NAS under:
- `A7_*****/<client_name>/2022_0128_5-Objects-Data/` 
- `A7_*****/<client_name>/20211207_見直しデータ`

#### Preprocessing
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
The masks are 8 by 5 pixels since this is the label precision we got.

Generate segmentation masks with (make sure the classes and max danger level are set in the data config file):
```
python -m utils.create_segmentation_masks <path to the dataset> <path to where the stiched masks will be saved>
```
Example:
```
python -m utils.create_segmentation_masks ../data/20220128_Reviewed_OutsideOffice/ ../data/preprocessed
```
Note: You can add the `-f` option to generate full size masks. (Not necessary, but helps a lot for visualization / to check for bugs)

TODO: For now the danger level mask value is not changer. But maybe there should be a -1 to have the lowest level be 0.

#### Split the data
You need to split the data between two folders: "Train" and "Validation" (the names are hard coded). You can either do it by hand, or modify the `utils/split_train_val.py` script to work on your dataset.

TODO: finish removing references to the classes.json to use only the data_config. In the meantime, the order of the 2 needs to match.
You then need to create a classes.json next to the Train and Validation folder, with the names of the classes (one per line). (see [here](https://github.com/hoel-bagard/Segmentation-PyTorch/wiki/Dataset-Preprocessing) for an example)

## Train
Once you have the environment all set up and your two config files ready, training an AI is straight forward. Just connect to the server of your choice (make sure the dependencies are installed) and run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

Notes:
The whole code folder will be copied in the checkpoint directory, in order to always have code that goes with the checkpoints. This means that you should not put your virtualenv or checkpoint directory in the code folder.
`CUDA_VISIBLE_DEVICES` is used to select which GPU to use. Check that the one you plan to use is free before you use it by running the `nvidia-smi` command.

## Inference
Use `python inference.py --help` to see all the options. The `model_config.py` must correspond to the one used when training the checkpoint.

Example:
```
python inference.py checkpoints/train_120.pt  ../data/preprocessed_data/Validation -j ../data/tiled_data/classes.json  -ts 1024 1024 -s 256 256 -d
```
The tile size should ideally be the same as used to cut the image before training. The strides can be whatever (although smaller than the tile size is better^^).

## Misc
### Formating
The code is trying to follow diverse PEPs conventions (notably PEP8). To have a similar dev environment you can install the following packages (pacman is for arch-based linux distros):

```
sudo pacman -S flake8 python-flake8-docstrings
pip install pep8-naming flake8-import-order
```
