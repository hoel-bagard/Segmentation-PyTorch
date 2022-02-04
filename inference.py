"""Inference script.

Basically the same as the test one, but handles tiling.
"""
import logging
from argparse import ArgumentParser
from json import load
from pathlib import Path
from typing import Optional

import albumentations
import cv2
import numpy as np
import torch
from einops import rearrange

from config.model_config import get_model_config
from src.dataset.data_transformations_albumentations import albumentation_wrapper
from src.dataset.dataset_specific_fn import default_get_mask_path as get_mask_path
from src.dataset.default_loader import (
    default_load_data,
    default_load_labels,
    default_loader
)
from src.networks.build_network import build_model
from src.torch_utils.utils.logger import create_logger
from src.torch_utils.utils.misc import get_dataclass_as_dict, show_img


def draw_blobs(img: torch.Tensor,
               mask_pred: np.ndarray,
               mask_label: np.ndarray,
               keypoints_pred,
               size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Place the segmentation masks next to the original image. Also puts the predicted blobs on the original image.

    Args:
        img: Original image
        mask_pred: RGB segmentation map predicted by the network
        mask_label: RGB label segmentation mask
        keypoints_pred: Blobs from the opencv blob detector
        size: If given, the images will be resized to this size

    Returns:
        (np.ndarray): RGB segmentation masks and original image (in one image)
    """
    img = rearrange(img, "c w h -> w h c").cpu().detach().numpy().astype(np.uint8)
    width, height, _ = img.shape

    # Create a blank image with some text to explain what things are
    text_img = np.full((width, height, 3), 255, dtype=np.uint8)
    text_img = cv2.putText(text_img, "Top left: input image.", (20, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Top right: label mask", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Bottom left: predicted mask", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    img_with_detections = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
                                            (255, 0, 0),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    out_img_top = cv2.hconcat((img_with_detections, mask_label))
    out_img_bot = cv2.hconcat((mask_pred, text_img))
    out_img = cv2.vconcat((out_img_top, out_img_bot))

    if size is not None:
        out_img = cv2.resize(out_img, size, interpolation=cv2.INTER_AREA)

    return out_img


def get_label_maps(classes_json_path: Path, logger: logging.Logger) -> tuple[dict[int, str], np.ndarray]:
    """Create 'maps' linking an int (class nb) to either a class name or its color.

    Args:
        classes_json: Path to a json with a list of {"name": "class_name", "color": [R,G,B]}
        logger: Logger used to print things.

    Returns:
        A dictionary mapping an int to a class name, and a list of colors (index is the class nb)
    """
    # Create the map between label (int) and color.
    assert classes_json_path.exists(), "\nCould not find the classes.json file"
    label_map = {}   # Maps an int to a class name
    color_map = []   # Maps an int to a color (corresponding to a class)
    with open(classes_json_path) as json_file:
        data = load(json_file)
        for key, entry in enumerate(data):
            label_map[key] = entry["name"]
            color_map.append(entry["color"])
    color_map = np.asarray(color_map)
    logger.info("Color map loaded")
    return label_map, color_map


def main():
    parser = ArgumentParser(description="Segmentation inference")
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the test dataset")
    parser.add_argument("--json_path", "-j", type=Path,
                        help="Json file with the index mapping, defaults to data_path.parent / 'classes.json'")
    parser.add_argument("--tile_size", "-ts", nargs=2, default=[256, 256], type=int, help="Size of the tiles (w, h)")
    parser.add_argument("--stride", "-s", nargs=2, default=[100, 100], type=int, help="Strides (w, h)")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    model_path: Path = args.model_path
    data_path: Path = args.data_path
    json_path: Path = args.json_path
    tile_width: int
    tile_height: int
    tile_width, tile_height = args.tile_size
    stride_width: int
    stride_height: int
    stride_width, stride_height = args.stride
    verbose_level: str = args.verbose_level

    model_config = get_model_config()
    logger = create_logger("Inference", verbose_level=verbose_level)
    label_map, color_map = get_label_maps(json_path if json_path else data_path.parent / "classes.json", logger)

    imgs_paths, masks_paths = default_loader(data_path, get_mask_path_fn=get_mask_path, verbose=False)
    assert len(imgs_paths) > 0, f"Did not find any image in {data_path}, exiting"
    logger.info(f"Data loaded, found {(nb_imgs := len(imgs_paths))} images.")

    preprocess_fn = albumentations.Compose([
        albumentations.Normalize(mean=model_config.MEAN, std=model_config.STD, max_pixel_value=255.0, p=1.0),
        albumentations.Resize(*model_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR)
    ])
    resize_to_original = albumentations.Resize(tile_height, tile_width, interpolation=cv2.INTER_LINEAR)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create and load the model
    print("Building model. . .", end="\r")
    model = build_model(model_config.MODEL, len(label_map), model_path=model_path,
                        eval_mode=True, **get_dataclass_as_dict(model_config))
    logger.info("Weights loaded     ")

    for i, (img_path, mask_path) in enumerate(zip(imgs_paths, masks_paths)):
        logger.debug(f"Processing image {img_path.name} ({i+1}/{nb_imgs})")

        img = default_load_data(img_path)
        height, width, _ = img.shape
        one_hot_mask = default_load_labels(mask_path)
        assert one_hot_mask.shape[0] == height and one_hot_mask.shape[1] == width, (
            f"\nShape of the image and the mask do not match for image {img_path}")

        pred_mask = np.zeros_like(one_hot_mask)
        for x in range(0, width-tile_width, stride_width):
            for y in range(0, height-tile_height, stride_height):
                tile = img[y:y+tile_height, x:x+tile_width]

                with torch.no_grad():
                    tile = np.expand_dims(resized_tile := preprocess_fn(image=tile)["image"], axis=0)
                    tile = tile.transpose((0, 3, 1, 2))
                    tile = torch.from_numpy(tile).float().to(device)
                    oh_tile_pred = model(tile)

                oh_tile_pred = rearrange(oh_tile_pred, "b c w h -> b w h c")
                # TODO: Mult here to skew towards over-detection
                oh_tile_pred = np.squeeze(oh_tile_pred.cpu().detach().numpy(), axis=0)
                oh_tile_pred = resize_to_original(image=resized_tile, mask=oh_tile_pred)["mask"]

                pred_mask[y:y+tile_height, x:x+tile_width] += oh_tile_pred

        pred_mask = np.argmax(pred_mask, axis=-1)
        # Recreate the segmentation mask from its one hot representation
        label_mask_rgb = cv2.cvtColor(np.asarray(color_map[pred_mask], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        show_img(label_mask_rgb)




        # pred_mask_rgb = np.asarray(color_map[pred_mask], dtype=np.uint8)
        # label_mask_rgb = np.asarray(color_map[label_mask], dtype=np.uint8)



if __name__ == "__main__":
    main()
