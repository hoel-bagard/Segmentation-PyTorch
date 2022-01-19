import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    parser = ArgumentParser("Preprocesses the green project images: trims and rename them.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--output_path", "-o", type=Path, default=None,
                        help="Path to where the trimmed images will be saved.")
    parser.add_argument("--nb_images_val", "-v", type=int, default=1, help="Number of images to save for validation.")
    parser.add_argument("--merge_anomalies", "-m", action="store_true", help="Make blue defects red.")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path if args.output_path else data_path.parent / "green_p_trimmed"
    nb_val: int = args.nb_images_val
    merge_anomalies: bool = args.merge_anomalies
    debug: bool = args.debug

    mask_path_list = list(data_path.rglob("*.png"))
    img_path_list = list(data_path.rglob("*.tiff"))
    nb_imgs = len(img_path_list)
    assert len(mask_path_list) == nb_imgs, f"Found {nb_imgs} images but {len(mask_path_list)} masks."
    print(f"Found {nb_imgs} images to process, starting to convert them to true black/white")

    min_y_list: list[int] = []
    max_y_list: list[int] = []
    for i, img_path in enumerate(img_path_list, start=1):
        msg = f"Processing status: ({i}/{nb_imgs})     ({img_path})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

        # In the original data, the image is a .tiff, and the corresponding mask is a file with the same name
        # (and path )but with a .png extension.
        # Here I don't use the mask's path directly since I want to check that each image has a corresponding mask.
        mask_path = img_path.with_suffix(".png")
        assert mask_path.exists(), f"Could not find mask for image {img_path}"
        mask = cv2.imread(str(mask_path))

        # Look for the highest and lowest anomaly pixel on the mask (red or blue pixel).
        anomaly_indices = np.argwhere(np.all(mask == [0, 0, 255], axis=-1) | np.all(mask == [255, 0, 0], axis=-1))
        min_y = np.amin(anomaly_indices[:, 0])
        max_y = np.amax(anomaly_indices[:, 0])
        if debug:
            # In order to visualize them, draw a top and bottom lines and show the resulting image.
            height, width, _ = mask.shape
            debug_mask = cv2.line(mask, (0, min_y), (width, min_y), (255, 255, 255), 10)
            debug_mask = cv2.line(debug_mask, (0, max_y), (width, max_y), (255, 255, 255), 10)
            show_img(debug_mask)

        min_y_list.append(min_y)
        max_y_list.append(max_y)

    if debug:
        print(f"Min ys: {min_y_list}")
        print(f"Max ys: {max_y_list}")
    print("\nMax y stats:")
    print(f"\tMax: {np.amax(max_y_list)}, min: {np.amin(max_y_list)}, "
          f"average: {np.mean(max_y_list):.2f}, variance: {np.var(max_y_list):.2f}.")
    print("Min y stats:")
    print(f"\tMax: {np.amax(min_y_list)}, min: {np.amin(min_y_list)}, "
          f"average: {np.mean(min_y_list):.2f}, variance: {np.var(min_y_list):.2f}.")
    crop_max_y = np.amax(max_y_list)
    crop_min_y = np.amin(min_y_list)

    # Split data between train and split
    # This assumes that nb_val is somewhat small. It also assumes that bright and dark image pairs share the same name.
    # And that no other image has the same name.
    validation_names = list([img_path.name for img_path in img_path_list[:nb_val]])

    print(f"Now trimming images using the min and max y. Saving them in {output_path}")
    for i, img_path in enumerate(img_path_list, start=1):
        msg = f"Processing status: ({i}/{nb_imgs})     ({img_path})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(img_path.with_suffix(".png")))

        img = img[crop_min_y:crop_max_y]
        mask = mask[crop_min_y:crop_max_y]

        if merge_anomalies:
            blue_mask = np.all(mask == [255, 0, 0], axis=-1)
            mask[blue_mask] = [0, 0, 255]

        rel_path = img_path.relative_to(data_path)
        split = "Validation" if img_path.name in validation_names else "Train"
        output_img_path = output_path / split / rel_path.with_suffix(".jpg")
        output_mask_path = output_path / split / rel_path.parent / (rel_path.stem + "_mask.png")
        output_img_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_img_path), img)
        cv2.imwrite(str(output_mask_path), mask)

    print("\nPreprocessing finished!")


if __name__ == "__main__":
    main()
