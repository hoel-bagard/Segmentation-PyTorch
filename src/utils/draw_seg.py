from typing import Optional
from functools import partial

import cv2
import numpy as np
import numpy.typing as npt


def draw_segmentation_danger_p(imgs: npt.NDArray[np.uint8],
                               masks_labels: npt.NDArray[np.int64],
                               masks_preds: npt.NDArray[np.int64],
                               idx_to_color: npt.NDArray[np.uint8],
                               max_danger_lvl: int,
                               size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Recreate the segmentation masks from their one hot representations, and place them next to the original image.

    Args:
        input_imgs: Destandardized input images.
                    Images should have 2 channels: grayscale image and disparity map.
        masks_labels: Label segmentation masks (using the class index (i.e. post argmax)).
                      Masks should also have 2 channels: classification and danger level.
        masks_pred: Predicted masks (using the class index (i.e. post argmax)).
                    Same as for the labels, should also have 2 channels
        idx_to_color: Array linking a class index to its color.
        max_danger_level (int): Max value for the danger level, used to scale from index to color.
        size (int, optional): If given, the images will be resized to this size.

    Returns:
        np.ndarray: RGB segmentation masks and original image (in one image)
    """
    height, width, _ = imgs[0].shape
    mask_width, mask_height = width // 2, height // 2

    put_text = partial(cv2.putText, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0,),
                       thickness=1, lineType=cv2.LINE_AA)
    # Create a blank image with some text to explain what things are
    text_img = np.full((height, width, 3), 255, dtype=np.uint8)
    text_img = put_text(text_img, "Top left: input grayscale image.", (20, 20))
    text_img = put_text(text_img, "Top right: input disparity map", (20, 40))
    text_img = put_text(text_img, "Bottom left: the danger and classification masks", (20, 60))
    text_img = put_text(text_img, "Top masks: predicted (left) and label (right) classification masks", (20, 80))
    text_img = put_text(text_img, "Bottom masks: predicted (left) and label (right) danger masks", (20, 100))

    out_imgs = []
    for img, pred_mask, label_mask in zip(imgs, masks_preds, masks_labels):
        pred_cls_mask, pred_danger_mask = pred_mask[..., 0], pred_mask[..., 1]
        label_cls_mask, label_danger_mask = label_mask[..., 0], label_mask[..., 1]

        # Make the classification masks into RGB image for visualization.
        pred_cls_mask = np.asarray(idx_to_color[pred_cls_mask], dtype=np.uint8)
        label_cls_mask = np.asarray(idx_to_color[label_cls_mask], dtype=np.uint8)

        # For the danger level, just go from dark for safe to white for danger (high danger lvl)
        pred_danger_mask = (255 * pred_danger_mask / max_danger_lvl).astype(np.uint8)
        pred_danger_mask = cv2.cvtColor(pred_danger_mask, cv2.COLOR_GRAY2RGB)
        label_danger_mask = (255 * label_danger_mask / max_danger_lvl).astype(np.uint8)
        label_danger_mask = cv2.cvtColor(label_danger_mask, cv2.COLOR_GRAY2RGB)

        # Scale all the masks to fit them on the final image.
        # The Labels' width/height are the way they are to handle the (unlikely) case where the images have an odd size
        pred_cls_mask = cv2.resize(pred_cls_mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST_EXACT)
        label_cls_mask = cv2.resize(label_cls_mask, (width-mask_width, height-mask_height),
                                    interpolation=cv2.INTER_NEAREST_EXACT)
        pred_danger_mask = cv2.resize(pred_danger_mask, (mask_width, mask_height),
                                      interpolation=cv2.INTER_NEAREST_EXACT)
        label_danger_mask = cv2.resize(label_danger_mask, (width-mask_width, height-mask_height),
                                       interpolation=cv2.INTER_NEAREST_EXACT)

        # Combine all the masks into one image:
        cls_mask = cv2.hconcat((pred_cls_mask, label_cls_mask))
        danger_mask = cv2.hconcat((pred_danger_mask, label_danger_mask))
        masks = cv2.vconcat((cls_mask, danger_mask))

        grayscale_img = cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2RGB)  # The "classic" input image
        disp = cv2.cvtColor(img[..., 1], cv2.COLOR_GRAY2RGB)  # The disparity map

        out_img_top = cv2.hconcat((grayscale_img, disp))
        out_img_bot = cv2.hconcat((masks, text_img))
        out_img = cv2.vconcat((out_img_top, out_img_bot))
        if size:
            out_img = cv2.resize(out_img, size, interpolation=cv2.INTER_NEAREST_EXACT)
        out_imgs.append(out_img)

    return np.asarray(out_imgs)


if __name__ == "__main__":
    def _test_draw():
        from config.data_config import get_data_config
        from src.torch_utils.utils.imgs_misc import show_img

        data_config = get_data_config()
        put_text = partial(cv2.putText, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0,),
                           thickness=1, lineType=cv2.LINE_AA)

        width, height = 600, 400
        gray_img = np.full((height, width), 255, dtype=np.uint8)
        gray_img = put_text(gray_img, "Image", (width//2, height//2))
        disp = np.full((height, width), 255, dtype=np.uint8)
        disp = put_text(disp, "Disparity", (width//2, height//2))
        img = np.stack((gray_img, disp), axis=-1)

        masks = np.random.randint(4, size=(10, 16, 4))

        imgs = np.asarray([img, img])
        masks_labels = np.asarray([masks[..., 0:2]//2, masks[..., 0:2]//2])
        masks_preds = np.asarray([masks[..., 2:], masks[..., 2:]])
        out_imgs = draw_segmentation_danger_p(imgs, masks_labels, masks_preds, data_config.IDX_TO_COLOR, 4)

        show_img(out_imgs[0], "Draw danger p test")
    _test_draw()
