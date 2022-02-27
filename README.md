# Segmentation-PyTorch

Things I want to change:
- Clean the disparity maps as a preprocessing step.
- The loss. (although I haven't much experience with losses and tiny masks)


yay -S otf-takao


Code was not meant, at all, to handle a double segmentation task. Sorry =(

In the code, "classification" refers to the object/class classification part. And danger refers to the danger level classification.


Many functions in the code are from old projects and have not been adapted for this one, they are therefore not usuable (and not needed).


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

See the readme in the [dataset folder](src/dataset) to see how to format the data.

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
