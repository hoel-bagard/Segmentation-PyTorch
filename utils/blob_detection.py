import argparse
from pathlib import Path
from shutil import get_terminal_size

import cv2
import numpy as np


def show_image(img, title: str = "Image"):
    while True:
        cv2.imshow(title, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    parser = argparse.ArgumentParser("Cuts images and corresponding masks into small tiles")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--show_missed", "--s", action="store_true",
                        help="Show sample where the blob detection failed")
    parser.add_argument("--debug", "--d", action="store_true", help="Show every sample")
    args = parser.parse_args()

    data_path: Path = args.data_path

    pos_truths = 0.0
    neg_truths = 0.0
    pos_elts = 0
    neg_elts = 0

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 50000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = False
    params.minThreshold = 5
    params.maxThreshold = 250
    detector = cv2.SimpleBlobDetector_create(params)

    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(file_list)
    for i, img_path in enumerate(file_list):
        msg = f"Processing image {img_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

        img = cv2.imread(str(img_path), 0)  # 0 to read in grayscale mode

        if args.debug:
            show_image(img, "Before transformations")

        # Differentiation tests
        # TODO: Has potential to be fine tuned through grid search.
        img = cv2.medianBlur(img, 7)
        ddepth = cv2.CV_64F  # cv2.CV_8U
        ksize, scale, delta = 3, 3, 3  # Defaults: 3, 1, 0
        grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Adaptive Thresholding tests  (https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html)
        # Increases detection rate, but also increases false positive.
        # TODO: Has potential to be fine tuned through grid search.
        # img = cv2.medianBlur(img, 7)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)

        # Sharpening tests
        # (Also did some quantization test much success)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Test1: doesn't change anything
        # img = np.clip(1.5*img - 0.5*cv2.GaussianBlur(img, (5, 5), 0), 0, 255).astype(np.uint8)
        # Test2: made it worse
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # neg_v = -0.5
        # kernel = np.array([[neg_v, neg_v, neg_v],
        #                     [neg_v, 1-8*neg_v, neg_v],
        #                     [neg_v, neg_v, neg_v]])
        # img = cv2.filter2D(img, -1, kernel)
        # Test3: Tried blurring the image, made it worse
        # img = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)

        if args.debug:
            show_image(img, "After transformations")
            # continue

        # Run the blob detector on the image and store the results
        keypoints_pred = detector.detect(img)
        if "bad" in str(img_path):  # or "ng" in img_path.stem:
            if len(keypoints_pred) > 0:
                neg_truths += 1
            elif args.show_missed:
                img_with_detection = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
                                                       (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                show_image(img_with_detection, "Sample with missed defect")
            neg_elts += 1
        else:
            if len(keypoints_pred) == 0:
                pos_truths += 1
            elif args.show_missed:
                img_with_detection = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
                                                       (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                show_image(img_with_detection, "Clean sample misclassified")
            pos_elts += 1

    print("\nFinished processing dataset")
    print(f"Processed {pos_elts} good samples and {neg_elts} bad samples")
    print(f"Positive accuracy: {pos_truths / pos_elts}, defect accuracy: {neg_truths / neg_elts}")
    print(f"Average accuracy: {(pos_truths + neg_truths) / (pos_elts + neg_elts)}")


if __name__ == "__main__":
    main()
