from argparse import ArgumentParser
from pathlib import Path

import albumentations
import cv2
import numpy as np
import onnxruntime as nxrun
import torch
from einops import rearrange

from config.data_config import get_data_config
from config.model_config import get_model_config
from src.dataset.danger_p_loader import danger_p_load_data as load_data
from src.dataset.danger_p_loader import danger_p_load_labels as load_labels
from src.dataset.danger_p_loader import danger_p_loader as data_loader
from src.dataset.data_transformations_albumentations import albumentation_wrapper
from src.dataset.dataset_specific_fn import default_get_mask_path
from src.torch_utils.utils.imgs_misc import denormalize_np
from src.torch_utils.utils.imgs_misc import show_img
from src.torch_utils.utils.logger import create_logger
from src.utils.draw_seg import draw_segmentation_danger_p


def main():
    parser = ArgumentParser(description="Inference Script")
    parser.add_argument("model_path", type=Path, help="Path to the onnx checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the test dataset")
    parser.add_argument("--output_path", "-o", type=Path, default=None, help="Save results to that folder if given.")
    parser.add_argument("--display_image", "-d", action="store_true", help="Show result.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    model_path: Path = args.model_path
    data_path: Path = args.data_path
    output_folder: Path = args.output_path
    display_img: bool = args.display_image
    verbose_level: str = args.verbose_level

    model_config = get_model_config()
    data_config = get_data_config()
    logger = create_logger("Inference", verbose_level=verbose_level)

    imgs_paths, masks_paths = data_loader(data_path, get_mask_path_fn=default_get_mask_path, verbose=False)
    assert len(imgs_paths) > 0, f"Did not find any image in {data_path}, exiting"
    logger.info(f"Data loaded, found {(nb_imgs := len(imgs_paths))} images.")

    preprocess_fn = albumentation_wrapper(albumentations.Compose([
        albumentations.Normalize(mean=model_config.MEAN, std=model_config.STD, max_pixel_value=255.0, p=1.0),
        albumentations.Resize(*model_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR)
    ]))

    print("Building model. . .", end="\r")
    model = nxrun.InferenceSession(model_path)
    input_name = model.get_inputs()[0].name
    logger.info("Weights loaded. Starting to process images (this might take a while).")

    for i, (img_path, mask_path) in enumerate(zip(imgs_paths, masks_paths)):
        logger.info(f"Processing image {img_path.name} ({i+1}/{nb_imgs})")

        img = load_data(img_path)
        img = preprocess_fn(np.expand_dims(img, 0))  # TODO: Label missing.
        one_hot_mask_label = load_labels(mask_path)

        cls_oh_preds, danger_oh_preds = model.run(None, {input_name: img.numpy()})

        img = rearrange(img, "b c w h -> b w h c").numpy().squeeze()
        imgs_batch = denormalize_np(img)

        cls_mask_label = torch.argmax(one_hot_mask_label[..., 0], dim=-1).cpu().detach().numpy()
        danger_mask_label = torch.argmax(one_hot_mask_label[..., 1], dim=-1).cpu().detach().numpy()

        cls_oh_preds = rearrange(cls_oh_preds, "b c w h -> b w h c")
        cls_masks_preds = torch.argmax(cls_oh_preds, dim=-1).cpu().detach().numpy()
        danger_oh_preds = rearrange(danger_oh_preds, "b c w h -> b w h c")
        danger_masks_preds = torch.argmax(danger_oh_preds, dim=-1).cpu().detach().numpy()

        masks_labels = np.stack((cls_mask_label, danger_mask_label), axis=-1)
        masks_preds = np.stack((cls_masks_preds, danger_masks_preds), axis=-1)
        out_img = draw_segmentation_danger_p(imgs_batch,
                                             masks_labels,
                                             masks_preds,
                                             data_config.IDX_TO_COLOR,
                                             data_config.MAX_DANGER_LEVEL)
        if display_img:
            show_img(out_img)
        if output_folder:
            rel_path = img_path.relative_to(data_path)
            output_path = output_folder / rel_path.parent / img_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), out_img)


if __name__ == "__main__":
    main()
