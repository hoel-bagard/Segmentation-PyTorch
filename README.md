# Segmentation-PyTorch
## Installation

### Dependencies
python version >= 3.10\
torch\
torchsummary\
tensorboard\
opencv-python

### Clone the repository
```
git clone git@github.com:hoel-bagard/Segmentation-PyTorch.git --recurse-submodules
```

### Get some data and format it:
You can get some data on the NAS (I used the "20211224\_Latest" data).

Place the data on your PC / server, and then use the following command to trim and rename the images:
```bash
python utils/green_preprocessing.py <path to data> -o <path to output folder> -m
```
For example:
```bash
python utils/green_preprocessing.py ../data/original_data/ -o ../data/preprocessed_data -m
```

Then tile the images with:
```bash
python utils/cut_images.py <path to preprocessed data> <path to output folder> -ts 512 512 -s 256 256
```
For example:
```bash
python utils/cut_images.py ../data/preprocessed_data/ ../data/tiled_data -ts 512 512 -s 256 256
```

Finally, you then need to create a classes.names next to the Train and Validation folder, with the names of the classes (one per line).

## Config files
In the config folder of this repo you will find two config template files. You need to copy them and remove the "_template" part.

### DataConfig
Contains most of the parameters regarding the data. Most of the values in the template can be kept as they are. The 3 paths usually need to be modified for each training (`DATA_PATH`, `CHECKPOINT_DIR` & `TB_DIR`). 

### ModelConfig
Contains the parameters that influence training. The default values should work fine, but you can try to tweak them to get better results. For the `MAX_EPOCHS` value, usually around 400 or 600 epochs is enough, you will need to train at least once to get an idea for your particular dataset.

## Train
Once you have the environment all set up and your two config files ready, training an AI is straight forward. Just connect to the server of your choice (make sure the dependencies are installed) and run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

Notes:
The whole code folder will be copied in the checkpoint directory, in order to always have code that goes with the checkpoints. This means that you should not put your virtualenv or checkpoint directory in the code folder.
`CUDA_VISIBLE_DEVICES` is used to select with GPU to use. Check that the one you plan to use is free before you use it by running the `nvidia-smi` command.


## Misc
### Formating
The code is trying to follow diverse PEPs conventions (notably PEP8). To have a similar dev environment you can install the following packages (pacman is for arch-based linux distros):

```
sudo pacman -S flake8 python-flake8-docstrings
pip install pep8-naming flake8-import-order
```

(The docstrings follow the google format.)
