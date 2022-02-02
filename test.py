import logging
from argparse import ArgumentParser
from functools import partial
from json import load
from pathlib import Path
from typing import Optional

import albumentations
import cv2
import numpy as np
import torch
from einops import rearrange

import src.dataset.data_transformations as transforms
from config.model_config import get_model_config
from src.dataset.data_transformations_albumentations import albumentation_wrapper
from src.dataset.dataset_specific_fn import default_get_mask_path as get_mask_path
from src.dataset.default_loader import (
    default_load_data,
    default_load_labels,
    default_loader
)
from src.networks.build_network import build_model
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.classification_metrics import ClassificationMetrics
from src.torch_utils.utils.draw import denormalize_tensor, draw_segmentation
from src.torch_utils.utils.logger import create_logger
from src.torch_utils.utils.misc import get_dataclass_as_dict


def show_image(img, title: str = "Image", already_bgr: bool = False):
    if not already_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    while True:
        cv2.imshow(title, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


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
    parser.add_argument("--json_path", "-j", type=Path, help="Path to the classes.json file")
    parser.add_argument("--show_imgs", "-s", action="store_true", help="Show predicted segmentation masks")
    parser.add_argument("--use_blob_detection", "-b", action="store_true",
                        help="Use blob detection on predicted masks to get a binary classification")
    parser.add_argument("--show_missed", "-sm", action="store_true",
                        help="Show samples where the blob detection failed")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    model_path: Path = args.model_path
    data_path: Path = args.data_path
    json_path: Path = args.json_path
    verbose_level: str = args.verbose_level
    show_imgs: bool = args.show_imgs
    show_missed: bool = args.show_missed
    use_blob_detection: bool = args.use_blob_detection

    model_config = get_model_config()
    logger = create_logger("Inference", verbose_level=verbose_level)

    label_map, color_map = get_label_maps(json_path if json_path else data_path.parent / "classes.json", logger)
    denormalize_img_fn = partial(denormalize_tensor,
                                 mean=torch.Tensor(model_config.MEAN),
                                 std=torch.Tensor(model_config.STD))

    data, labels = default_loader(data_path, get_mask_path_fn=get_mask_path, verbose=False)
    assert len(labels) > 0, f"Did not find any image in {data_path}, exiting"
    logger.info(f"Data loaded, found {len(labels)} images.")

    common_pipeline = albumentation_wrapper(albumentations.Compose([
        albumentations.Normalize(mean=model_config.MEAN, std=model_config.STD, max_pixel_value=255.0, p=1.0),
        albumentations.Resize(*model_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR)
    ]))
    with BatchGenerator(data, labels, 1,
                        nb_workers=2,
                        data_preprocessing_fn=default_load_data,
                        labels_preprocessing_fn=default_load_labels,
                        cpu_pipeline=common_pipeline,
                        gpu_pipeline=transforms.to_tensor(),
                        shuffle=False) as dataloader:
        logger.debug("Dataloader created")

        # Creates and load the model
        print("Building model. . .", end="\r")
        model = build_model(model_config.MODEL, len(label_map), model_path=model_path,
                            eval_mode=True, **get_dataclass_as_dict(model_config))
        logger.info("Weights loaded     ")

        if use_blob_detection:
            # Variables used to keep track of the classification results
            true_negs = 0.0
            true_pos = 0.0
            pos_elts = 0
            neg_elts = 0

            # Setup SimpleBlobDetector parameters.
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 1
            params.maxArea = 5000
            params.minThreshold = 0
            params.maxThreshold = 255
            params.filterByCircularity = False
            params.filterByColor = False
            params.filterByConvexity = False
            params.filterByInertia = False

            detector: cv2.SimpleBlobDetector = cv2.SimpleBlobDetector_create(params)

        with torch.no_grad():
            # Compute some segmentation metrics
            logger.info("Computing the confusion matrix")
            metrics = ClassificationMetrics(model, None, dataloader, label_map, max_batches=None, segmentation=True)
            metrics.compute_confusion_matrix(mode="Validation")
            avg_acc = metrics.get_avg_acc()
            logger.info(f"Average accuracy: {avg_acc}")

            per_class_acc = metrics.get_class_accuracy()
            per_class_acc_msg = ["\n\t" + label_map[key] + f": {acc}" for key, acc in enumerate(per_class_acc)]
            logger.info("Per Class Accuracy:" + "".join(per_class_acc_msg))

            per_class_iou = metrics.get_class_iou()
            per_class_iou_msg = ["\n\t" + label_map[key] + f": {iou}" for key, iou in enumerate(per_class_iou)]
            logger.info("Per Class IOU:" + "".join(per_class_iou_msg))

            if show_imgs:
                confusion_matrix = metrics.get_confusion_matrix()
                show_image(confusion_matrix, "Confusion Matrix")

            # Redo a pass over the dataset to get more information if requested
            if show_imgs or use_blob_detection:
                for _step, (inputs, labels) in enumerate(dataloader, start=1):
                    predictions = model(inputs)
                    inputs = denormalize_img_fn(inputs)

                    if use_blob_detection:
                        one_hot_masks_preds = rearrange(predictions, "b c w h -> b w h c")
                        masks_preds: np.ndarray = torch.argmax(one_hot_masks_preds, dim=-1).cpu().detach().numpy()
                        one_hot_masks_labels = rearrange(labels, "b c w h -> b w h c")
                        masks_labels: np.ndarray = torch.argmax(one_hot_masks_labels, dim=-1).cpu().detach().numpy()

                        width, height, _ = one_hot_masks_preds[0].shape
                        for img, pred_mask, label_mask in zip(inputs, masks_preds, masks_labels):
                            # Recreate the segmentation mask from its one hot representation
                            pred_mask_rgb = np.asarray(color_map[pred_mask], dtype=np.uint8)
                            label_mask_rgb = np.asarray(color_map[label_mask], dtype=np.uint8)

                            # Run the blob detector on the image and store the results
                            keypoints_pred = detector.detect(pred_mask_rgb)
                            keypoints_label = detector.detect(label_mask_rgb)
                            if len(keypoints_label) > 0:
                                if len(keypoints_pred) > 0:
                                    true_negs += 1
                                elif show_missed:
                                    out_img = draw_blobs(img, pred_mask_rgb, label_mask_rgb, keypoints_pred)
                                    show_image(out_img, "Sample with missed defect")
                                neg_elts += 1
                            else:
                                if len(keypoints_pred) == 0:
                                    true_pos += 1
                                elif show_missed:
                                    out_img = draw_blobs(img, pred_mask_rgb, label_mask_rgb, keypoints_pred)
                                    show_image(out_img, "Clean sample misclassified")
                                pos_elts += 1
                            if show_imgs:
                                out_img = draw_blobs(img, pred_mask_rgb, label_mask_rgb, keypoints_pred)
                                show_image(out_img, "Output image")
                    elif show_imgs:
                        out_imgs = draw_segmentation(inputs, predictions, labels, color_map=color_map)
                        for out_img in out_imgs:
                            show_image(out_img)

    if use_blob_detection:
        precision = true_pos / max(1, (true_pos + (neg_elts-true_negs)))
        recall = true_pos / max(1, pos_elts)
        acc = (true_pos + true_negs) / (neg_elts + pos_elts)
        pos_acc = true_pos / max(1, pos_elts)
        neg_acc = true_negs / max(1, neg_elts)

        stats = (precision, recall, acc, pos_acc, neg_acc)
        stats_names = ("Precision", "Recall", "Accuracy", "Positive accuracy", "Negative accuracy")

        logger.info(f"Dataset was composed of {pos_elts} good samples and {neg_elts} bad samples.")
        logger.info("Results obtained using blob detection for classification:")
        logger.info(f"Bad samples misclassified: {neg_elts-true_negs}, Good samples misclassified: {pos_elts-true_pos}")
        for stat_idx in range(len(stats)):
            logger.info(f"{stats_names[stat_idx]}: {stats[stat_idx]:.2f}")


if __name__ == "__main__":
    main()
