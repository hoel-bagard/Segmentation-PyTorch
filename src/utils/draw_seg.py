from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt


def draw_segmentation(imgs: npt.NDArray[np.uint8],
                      masks_labels: npt.NDArray[np.int64],
                      masks_preds: npt.NDArray[np.int64],
                      color_map: list[tuple[int, int, int]],
                      size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Recreate the segmentation masks from their one hot representations, and place them next to the original image.

    Args:
        input_imgs: Destandardized input images.
        masks_labels: Label segmentation masks (using the class index (i.e. post argmax)).
        masks_pred: Predicted masks (using the class index (i.e. post argmax)).
        color_map (list): List linking class index to its color
        size (int, optional): If given, the images will be resized to this size

    Returns:
        np.ndarray: RGB segmentation masks and original image (in one image)
    """
    width, height, _ = imgs[0].shape

    # Create a blank image with some text to explain what things are
    text_img = np.full((width, height, 3), 255, dtype=np.uint8)
    text_img = cv2.putText(text_img, "Top left: input image.", (20, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Top right: label mask", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Bottom left: predicted mask", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    out_imgs = []
    for img, pred_mask, label_mask in zip(imgs, masks_preds, masks_labels):
        # Recreate the segmentation mask from its one hot representation
        pred_mask_rgb = np.asarray(color_map[pred_mask], dtype=np.uint8)
        label_mask_rgb = np.asarray(color_map[label_mask], dtype=np.uint8)

        out_img_top = cv2.hconcat((img, label_mask_rgb))
        out_img_bot = cv2.hconcat((pred_mask_rgb, text_img))
        out_img = cv2.vconcat((out_img_top, out_img_bot))
        if size:
            out_img = cv2.resize(out_img, size, interpolation=cv2.INTER_NEAREST_EXACT)
        out_imgs.append(out_img)

    return np.asarray(out_imgs)
