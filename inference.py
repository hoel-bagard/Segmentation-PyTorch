"""Inference script.

Basically the same as the test one, but handles tiling, and uses connected components instead of blobs.

Note:
TODO Due to the way the for loops for tiling are written, it is not guaranteed that the whole image will be processed.
     (The end borders might not be in a tile. Remove the -tile_size in the range, add an if.)
"""
import logging
from argparse import ArgumentParser
from json import load
from pathlib import Path

import albumentations
import cv2
import numpy as np
import torch
from einops import rearrange

from config.model_config import get_model_config
from src.dataset.dataset_specific_fn import default_get_mask_path as get_mask_path
from src.dataset.default_loader import (
    default_load_data,
    default_load_labels,
    default_loader
)
from src.networks.build_network import build_model
from src.torch_utils.utils.imgs_misc import show_img
from src.torch_utils.utils.logger import create_logger
from src.torch_utils.utils.misc import get_dataclass_as_dict
from src.utils.iou import get_iou_bboxes, get_iou_masks


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


def draw_blobs_from_bboxes(img: np.ndarray,
                           bboxes: list[tuple[int, int, int, int]],
                           color: tuple[int, int, int]) -> np.ndarray:
    """Utils function that draws bounding boxes as ellipses."""
    img = img.copy()
    for left, top, width, height in bboxes:
        center = (left + width//2, top + height//2)
        img = cv2.ellipse(img, center, (width, height), 0, 0, 360, color=color, thickness=4)
    return img


def concat_imgs(img: np.ndarray,
                mask_pred: np.ndarray,
                mask_label: np.ndarray) -> np.ndarray:
    """Place the segmentation masks next to the original image.

    Args:
        img: Original image
        mask_pred: RGB segmentation map predicted by the network
        mask_label: RGB label segmentation mask

    Returns:
        RGB segmentation masks and original image (in one image)
    """
    height, width, _ = img.shape
    vert_stack = width > 4*height  # If the image is very wide, stack everything vertically instead of in a square.

    # Create a blank image with some text to explain what things are
    text_img = np.full((height, width, 3), 255, dtype=np.uint8)
    font_size = max(1, width//2000)
    text_img = cv2.putText(text_img, "Top: input image" if vert_stack else "Top left: input image.",
                           (20, 40*font_size), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1*font_size,
                           cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Middle: predicted mask" if vert_stack else "Top right: label mask",
                           (20, 80*font_size), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1*font_size,
                           cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Bottom: label mask" if vert_stack else "Bottom left: predicted mask",
                           (20, 120*font_size), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1*font_size,
                           cv2.LINE_AA)

    if vert_stack:
        out_img = cv2.vconcat((img, mask_pred, mask_label, text_img))
    else:
        out_img_top = cv2.hconcat((img, mask_label))
        out_img_bot = cv2.hconcat((mask_pred, text_img))
        out_img = cv2.vconcat((out_img_top, out_img_bot))

    return out_img


def get_cc_bboxes(img: np.ndarray, logger: logging.Logger, area_threshold: int = 0) -> list[tuple[int, int, int, int]]:
    """Compute the connected components on the given image and return their bounding boxes.

    Args:
        img: The image to process.
        logger: Used to print things if necessary.
        area_threshold: Can be used to filter out small components.

    Returns:
        A list of bounding bounding boxes (a bounding box being a tuple of (left, top, width, height))
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.connectedComponentsWithStatsWithAlgorithm
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    logger.debug(f"Found {nb_labels-1} connected components.")

    # Filter out small areas
    bboxes: list[tuple[int, int, int, int]] = []
    for (left, top, width, height, area) in stats:
        if area > area_threshold:
            bboxes.append((left, top, width, height))
    return bboxes


def get_confusion_matrix_from_bboxes(labels: list[tuple[int, int, int, int]],
                                     preds: list[tuple[int, int, int, int]]) -> tuple[int, int, int]:
    """Compute the True Positives, False Positives and False Negatives corresponding to the given labels / predictions.

    Notes: Does not take into account classes.
           A label can be "matched" (counted as a TP) only once. If multiple predictions match with the same label,
           only the first one will count. The other ones will be discarded.

    TODO: Add a named tuple BBox  (to make things more readable)

    Args:
        labels: The labels bounding boxes.
        preds: The predicted bounding boxes.

    Returns:
        3 ints, the TP, FP and FN numbers of occurence.
    """
    tp, fp, fn = 0, 0, 0,
    already_matched = []
    for bbox in preds:
        matched_idx = match_bboxes(bbox, labels)
        if matched_idx == -1:
            fp += 1
        elif matched_idx not in already_matched:
            already_matched.append(matched_idx)
            tp += 1
    fn = len(set(np.arange(len(labels))) - set(already_matched))

    return tp, fp, fn


def match_bboxes(bbox: tuple[int, int, int, int],
                 bboxes: list[tuple[int, int, int, int]],
                 iou_threshold: float = 0.) -> int:
    """Checks if the given bounding box overlapps with another from the list, if yes returns its index."""
    for i, target_bbox in enumerate(bboxes):
        if get_iou_bboxes(bbox, target_bbox) > iou_threshold:
            return i
    return -1


def main():
    parser = ArgumentParser(description="Segmentation inference")
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the test dataset")
    parser.add_argument("--output_path", "-o", type=Path, default=None, help="Save results to that folder if given.")
    parser.add_argument("--json_path", "-j", type=Path,
                        help="Json file with the index mapping, defaults to data_path.parent / 'classes.json'")
    parser.add_argument("--tile_size", "-ts", nargs=2, default=[256, 256], type=int, help="Size of the tiles (w, h)")
    parser.add_argument("--stride", "-s", nargs=2, default=[100, 100], type=int, help="Strides (w, h)")
    parser.add_argument("--display_image", "-d", action="store_true", help="Show result.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    model_path: Path = args.model_path
    data_path: Path = args.data_path
    output_folder: Path = args.output_path
    json_path: Path = args.json_path
    tile_width: int
    tile_height: int
    tile_width, tile_height = args.tile_size
    stride_width: int
    stride_height: int
    stride_width, stride_height = args.stride
    display_img: bool = args.display_image
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
    logger.info("Weights loaded. Starting to process images (this might take a while).")

    for i, (img_path, mask_path) in enumerate(zip(imgs_paths, masks_paths)):
        logger.info(f"Processing image {img_path.name} ({i+1}/{nb_imgs})")

        img = default_load_data(img_path)
        one_hot_mask = default_load_labels(mask_path)  # TODO: Make this step optional ?
        height, width, _ = img.shape
        assert one_hot_mask.shape[0] == height and one_hot_mask.shape[1] == width, (
            f"\nShape of the image and the mask do not match for image {img_path}")

        nb_tiles_per_img = (1 + (width-tile_width) // stride_width) * (1 + (height-tile_height) // stride_height)
        tile_idx = 0
        pred_mask = np.zeros_like(one_hot_mask)
        for x in range(0, width-tile_width, stride_width):
            for y in range(0, height-tile_height, stride_height):
                logger.debug(f"Processing tile {(tile_idx := tile_idx + 1)} / {nb_tiles_per_img}")
                tile = img[y:y+tile_height, x:x+tile_width]

                with torch.no_grad():
                    tile = np.expand_dims(resized_tile := preprocess_fn(image=tile)["image"], axis=0)
                    tile = tile.transpose((0, 3, 1, 2))
                    tile = torch.from_numpy(tile).float().to(device)
                    oh_tile_pred = model(tile)

                oh_tile_pred = rearrange(oh_tile_pred, "b c w h -> b w h c")
                oh_tile_pred = np.squeeze(oh_tile_pred.cpu().detach().numpy(), axis=0)
                oh_tile_pred = resize_to_original(image=resized_tile, mask=oh_tile_pred)["mask"]

                # Effectively averages the predictions from overlapping tiles.
                pred_mask[y:y+tile_height, x:x+tile_width] += oh_tile_pred

        # Skew the results towards over-detection if desired. Needs to be tweaked manually though
        # pred_mask[..., 1:] *= 4
        # Small post processing. Erode to remove small areas.
        pred_mask = cv2.GaussianBlur(pred_mask, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
        pred_mask = cv2.erode(pred_mask, kernel, iterations=1)
        # kernel = np.ones((3, 3), np.uint8)
        # pred_mask = cv2.dilate(pred_mask, kernel, iterations=1)

        # Go from logits to one hot
        pred_mask = np.argmax(pred_mask, axis=-1)
        # Recreate the segmentation mask from its one hot representation
        pred_mask_rgb = cv2.cvtColor(np.asarray(color_map[pred_mask], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        label_mask = cv2.imread(str(mask_path))

        label_bboxes = get_cc_bboxes(label_mask, logger)
        # Again, remove small predicted areas (tweak value depending on the project).
        pred_bboxes = get_cc_bboxes(pred_mask_rgb, logger, area_threshold=70)

        if output_folder:
            rel_path = img_path.relative_to(data_path)
            output_path = output_folder / rel_path.parent / img_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            drawn_img = draw_blobs_from_bboxes(img, pred_bboxes, (0, 0, 255))
            logger.info(f"Saving result image at {output_path}")
            cv2.imwrite(str(output_path), drawn_img)
        if (output_folder and logger.getEffectiveLevel() == logging.DEBUG) or display_img:
            drawn_img = draw_blobs_from_bboxes(img, label_bboxes, (0, 255, 0))
            drawn_img = draw_blobs_from_bboxes(drawn_img, pred_bboxes, (0, 0, 255))
            result_img = concat_imgs(drawn_img, pred_mask_rgb, label_mask)
            if display_img:
                show_img(result_img)
            if output_folder:
                output_path = output_path.with_stem(img_path.stem + "_debug")
                cv2.imwrite(str(output_path), result_img)

        tp, fp, fn = get_confusion_matrix_from_bboxes(label_bboxes, pred_bboxes)
        logger.info(f"Results for image {img_path}: TP: {tp}, FP: {fp}, FN: {fn}")

        iou = get_iou_masks(label_mask, pred_mask_rgb, color=(0, 0, 255))
        logger.info(f"\tIoU: {iou:.3f}")


if __name__ == "__main__":
    main()
