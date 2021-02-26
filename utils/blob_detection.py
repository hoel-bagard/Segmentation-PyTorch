import argparse
from pathlib import Path
from shutil import get_terminal_size

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser("Cuts images and corresponding masks into small tiles")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
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
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 5
    params.maxThreshold = 250
    detector = cv2.SimpleBlobDetector_create(params)

    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(file_list)
    for i, img_path in enumerate(file_list):
        msg = f"Processing image {img_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\n')

        img = cv2.imread(str(img_path))

        keypoints_pred = detector.detect(img)


        if "bad" in str(img_path) or "ng" in img_path.stem:
            if len(keypoints_pred) > 0:
                neg_truths += 1
                print(f"{img_path} bad detected")
            else:
                print(f"{img_path} bad not detected")
            neg_elts += 1
        else:
            if len(keypoints_pred) == 0:
                pos_truths += 1
                print(f"{img_path} good detected")
            else:
                print(f"{img_path} good not detected")
            pos_elts += 1





        # if len(keypoints_pred) > 0:
        #     img_with_detection = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
        #                                            (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #     while True:
        #         cv2.imshow("Frame", img_with_detection)
        #         key = cv2.waitKey(10)
        #         if key == ord("q"):
        #             cv2.destroyAllWindows()
        #             break

        # if len(keypoints_pred) > 0:
        #     img_with_detection = cv2.drawKeypoints(img, keypoints_pred, np.array([]),
        #                                            (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #     while True:
        #         cv2.imshow("Frame", img_with_detection)
        #         key = cv2.waitKey(10)
        #         if key == ord("q"):
        #             cv2.destroyAllWindows()
        #             break


    print("\nFinished processing dataset")
    print(f"Processed {pos_elts} good samples and {neg_elts} bad samples")
    print(f"Positive accuracy: {pos_truths / pos_elts}, negative accuracy: {neg_truths / neg_elts}")
    print(f"Average accuracy: {(pos_truths + neg_truths) / (pos_elts + neg_elts)}")


if __name__ == "__main__":
    main()
