from argparse import ArgumentParser
from pathlib import Path

import cv2
from einops import rearrange
import numpy as np
import torch

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.networks.build_network import build_model
from src.torch_utils.utils.misc import get_config_as_dict
import src.dataset.data_transformations as transforms
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.dataset.defeault_loader import (
    default_loader,
    default_load_data,
    default_load_labels
)
from src.dataset.dataset_specific_fn import get_mask_path_tape as get_mask_path
from src.torch_utils.utils.draw import draw_segmentation
from src.torch_utils.utils.metrics import Metrics


def show_image(img, title: str = "Image", already_rgb: bool = False):
    if not already_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    while True:
        cv2.imshow(title, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the test dataset")
    parser.add_argument("--show_imgs", "--s", action="store_true", help="Show predicted segmentation masks")
    parser.add_argument("--use_blob_detection", "--b", action="store_true",
                        help="Use blob detection on predicted masks to get a binary classification")
    parser.add_argument("--show_missed", "--sm", action="store_true",
                        help="Show samples where the blob detection failed")
    args = parser.parse_args()

    # Creates and load the model
    model = build_model(ModelConfig.MODEL, DataConfig.OUTPUT_CLASSES, model_path=args.model_path,
                        eval_mode=True, **get_config_as_dict(ModelConfig))
    print("Weights loaded", flush=True)

    # Create dataloader
    label_map = DataConfig.LABEL_MAP
    batch_size = 1
    base_gpu_pipeline = (transforms.to_tensor(), transforms.normalize(labels_too=True))
    data, labels = default_loader(args.data_path, get_mask_path_fn=get_mask_path)
    dataloader = BatchGenerator(data, labels, batch_size, nb_workers=DataConfig.NB_WORKERS,
                                data_preprocessing_fn=default_load_data,
                                labels_preprocessing_fn=default_load_labels,
                                gpu_augmentation_pipeline=transforms.compose_transformations(base_gpu_pipeline))

    if args.use_blob_detection:
        # Variables used to keep track of the classification results
        true_negs = 0.0
        true_pos = 0.0
        pos_elts = 0
        neg_elts = 0

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 50000
        params.minThreshold = 5
        params.maxThreshold = 250

        params.filterByCircularity = False
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)

    with torch.no_grad():
        # Compute some segmentation metrics
        metrics = Metrics(model, None, dataloader, DataConfig.LABEL_MAP, max_batches=None, segmentation=True)
        metrics.compute_confusion_matrix(mode="Validation")
        avg_acc = metrics.get_avg_acc()
        print(f"\nAverage accuracy: {avg_acc}")

        per_class_acc = metrics.get_class_accuracy()
        per_class_acc_msg = ["\n" + label_map[key] + f": {acc}" for key, acc in enumerate(per_class_acc)]
        print("\nPer Class Accuracy:" + "".join(per_class_acc_msg))

        per_class_iou = metrics.get_class_iou()
        per_class_iou_msg = ["\n" + label_map[key] + f": {iou}" for key, iou in enumerate(per_class_iou)]
        print("\nPer Class IOU:" + "".join(per_class_iou_msg))

        confusion_matrix = metrics.get_confusion_matrix()
        show_image(confusion_matrix, "Confusion Matrix")

        # Redo a pass over the dataset to get more information if requested
        if args.show_imgs or args.use_blob_detection:
            for step, (inputs, labels) in enumerate(dataloader, start=1):
                predictions = model(inputs)

                if args.show_imgs:
                    out_imgs = draw_segmentation(inputs, predictions, labels, color_map=DataConfig.COLOR_MAP)
                    for out_img in out_imgs:
                        show_image(out_img)
                if args.use_blob_detection:
                    one_hot_masks_preds = rearrange(predictions, "b c w h -> b w h c")
                    masks_preds: np.ndarray = torch.argmax(one_hot_masks_preds, dim=-1).cpu().detach().numpy()
                    one_hot_masks_labels = rearrange(labels, "b c w h -> b w h c")
                    masks_labels: np.ndarray = torch.argmax(one_hot_masks_labels, dim=-1).cpu().detach().numpy()

                    width, height, _ = one_hot_masks_preds[0].shape
                    for img, pred_mask, label_mask in zip(inputs, masks_preds, masks_labels):
                        # Recreate the segmentation mask from its one hot representation
                        pred_mask_rgb = np.empty((width, height, 3), dtype=np.uint8)
                        label_mask_rgb = np.empty((width, height, 3), dtype=np.uint8)
                        # TODO: optimize this later
                        for i in range(width):
                            for j in range(height):
                                pred_mask_rgb[i, j] = DataConfig.COLOR_MAP[pred_mask[i, j]]
                                label_mask_rgb[i, j] = DataConfig.COLOR_MAP[label_mask[i, j]]

                        # Run the blob detector on the image and store the results
                        keypoints_pred = detector.detect(pred_mask_rgb)
                        keypoints_label = detector.detect(label_mask_rgb)
                        if len(keypoints_label) > 0:
                            if len(keypoints_pred) > 0:
                                true_negs += 1
                            elif args.show_missed:
                                img = rearrange(img, "c w h -> w h c").cpu().detach().numpy()
                                img = np.asarray(img * 255.0, dtype=np.uint8)
                                img_with_detection = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
                                                                       (255, 0, 0),
                                                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                show_image(img_with_detection, "Sample with missed defect")
                            neg_elts += 1
                        else:
                            if len(keypoints_pred) == 0:
                                true_pos += 1
                            elif args.show_missed:
                                img = rearrange(img, "c w h -> w h c").cpu().detach().numpy()
                                img = np.asarray(img * 255.0, dtype=np.uint8)
                                img_with_detection = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
                                                                       (255, 0, 0),
                                                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                show_image(img_with_detection, "Clean sample misclassified")
                            pos_elts += 1

    if args.use_blob_detection:
        precision = true_pos / (true_pos + (neg_elts-true_negs))
        recall = true_pos / max(1, pos_elts)
        acc = (true_pos + true_negs) / (neg_elts + pos_elts)
        pos_acc = true_pos / max(1, pos_elts)
        neg_acc = true_negs / max(1, neg_elts)

        stats = (precision, recall, acc, pos_acc, neg_acc)
        stats_names = ("Precision", "Recall", "Accuracy", "Positive accuracy", "Negative accuracy")

        print(f"Dataset was composed of {pos_elts} good samples and {neg_elts} bad samples")
        print("\nResults obtained using blob detection for classification:")
        for stat_idx in range(len(stats)):
            print(f"{stats_names[stat_idx]}: {stats[stat_idx]}")


if __name__ == "__main__":
    main()
