import argparse
import glob
import os

import torch
from torchvision.transforms import Compose
import cv2

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.networks.build_network import build_model
import src.dataset.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the checkpoint to use")
    parser.add_argument("data_path", help="Path to the test dataset")
    args = parser.parse_args()

    # Creates and load the model
    model = build_model(ModelConfig.NETWORK, args.model_path, eval=True)
    print("Weights loaded", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_map = DataConfig.LABEL_MAP
    transform = Compose([
        transforms.Crop(top=600, bottom=500, left=800, right=200),
        transforms.Resize(*ModelConfig.IMAGE_SIZES),
        transforms.Normalize(),
        transforms.ToTensor()
    ])

    img_types = ("*.jpg", "*.bmp")
    for key in range(len(label_map)):
        pathname = os.path.join(args.data_path, label_map[key], "**")
        image_paths = []
        [image_paths.extend(glob.glob(os.path.join(pathname, ext), recursive=True)) for ext in img_types]
        for img_path in image_paths:
            msg = f"Loading data {img_path}"
            print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end="\r")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform({"img": img, "label": 0})["img"]   # The 0 is ignored
            img = img.unsqueeze(0).to(device).float()

            output = model(img)

            while True:
                cv2.imshow("Image", output)
                if cv2.waitKey(10) == ord("q"):
                    break


if __name__ == "__main__":
    main()
