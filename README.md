# Segmentation-PyTorch
## Installation

#### Clone the repository
```
git clone git@github.com:hoel-bagard/Segmentation-PyTorch.git --recurse-submodules
```

#### Japanese font
If using Japanese class names (this project is by default), you need to install the Takao font to render the confusion matrix. On Arch you can use `yay -S otf-takao` to do that. You can also edit the [metrics file](src/utils/seg_metrics.py) to change the font.

#### Dependencies
Install the dependencies. For example in a venv with `pip install -r requirements.txt`.

## Config files
In the config folder of this repo you will find two config template files. You need to copy them and remove the "_template" part.

#### DataConfig
Contains most of the parameters regarding the data. Most of the values in the template can be kept as they are. But some things need to be set:
- The class names and colors for the dataset (if a class is not included there, then it will be converted to "その他" when generating the masks).
- The 3 paths usually need to be modified for each training (`DATA_PATH`, `CHECKPOINT_DIR` & `TB_DIR`). 

Notes:
- The whole code folder will be copied in the checkpoint directory, in order to always have code that goes with the checkpoints. This means that you should not put your virtualenv or checkpoint directory in the code folder.


#### ModelConfig
Contains the parameters that influence training. The default values should work fine, but you can try to tweak them to get better results.\
For the `MAX_EPOCHS` value, usually around 400 epochs is enough, you will need to train at least once to get an idea for your particular dataset.

## Get some data and format it:

The data for this project is on the NAS under:
- `A7_*****/<client_name>/2022_0128_5-Objects-Data/` 
- `A7_*****/<client_name>/20211207_見直しデータ`

See the readme in the [dataset folder](src/dataset) to see how to format the data.

## Train
Once you have the environment all set up on the server of your choice and your two config files ready, training an AI is straight forward. Just run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python train.py --name <train_name>
```

Notes:
- Training should take around 1 hour.
- `CUDA_VISIBLE_DEVICES` is used to select which GPU to use. Check that the one you plan to use is free before you use it by running the `nvidia-smi` command.

#### Results
To see the results of the train, open the TensorBoard. You should see something similar to this in the image tab:

| Train Image 1 | Train Image 2 |
:-------------------------:|:-------------------------:
| ![1](https://user-images.githubusercontent.com/34478245/155912407-663943d7-c2f2-4d85-b698-d2942a04d39c.png) | ![2](https://user-images.githubusercontent.com/34478245/155912412-1e6c9f15-b4fb-4267-ba01-9de01cc3c7f8.png) |


There are also lots of plots and confusion matrices.

## Inference
There is no inference script for this project. At least not yet.

## TODOs
- Clean the disparity maps as a preprocessing step (make the white spots stand out less).
- Change the loss to the usual one ?
- More data augmentation.


## Misc
### Notes about the code
The code was not meant, at all, to handle a double segmentation task. It is therefore of rather poor quality. Sorry =(\
In the code, "classification" refers to the object/class classification part. And danger refers to the danger level classification.

### Formating
The code is trying to follow diverse PEPs conventions (notably PEP8). To have a similar dev environment you can install the following packages (pacman is for arch-based linux distros):
```
sudo pacman -S flake8 python-flake8-docstrings
pip install pep8-naming flake8-import-order
```
