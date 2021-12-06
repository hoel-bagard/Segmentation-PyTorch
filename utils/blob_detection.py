import os
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from shutil import get_terminal_size


import cv2
import numpy as np
from tqdm import tqdm


def grid_search_worker(args):
    file_list, filter_by_area, min_area, filter_by_circularity, min_circularity = args[:5]
    filter_by_convexity, min_treshold, max_treshold = args[5:]

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = filter_by_area
    params.minArea = float(min_area)
    params.maxArea = 50000
    params.filterByCircularity = filter_by_circularity
    params.minCircularity = min_circularity
    params.filterByConvexity = filter_by_convexity
    params.minConvexity = 0.5
    params.filterByInertia = False
    params.minThreshold = int(min_treshold)
    params.maxThreshold = int(max_treshold)
    detector = cv2.SimpleBlobDetector_create(params)

    # Variable used to store results
    true_negs = 0.0
    true_pos = 0.0

    for _, img_path in enumerate(file_list):
        img = cv2.imread(str(img_path), 0)  # 0 to read in grayscale mode
        # Run the blob detector on the image and store the result
        keypoints_pred = detector.detect(img)
        if "bad" in str(img_path):
            if len(keypoints_pred) > 0:
                true_negs += 1
        else:
            if len(keypoints_pred) == 0:
                true_pos += 1

    return (true_negs, true_pos)


def grid_search(data_path: Path):
    # Define all the parameters to search against
    # SimpleBlobDetector parameters
    filter_by_area = [True, False]
    min_area = np.arange(2, 50, 4)
    filter_by_circularity = [True, False]
    min_circularity = [0.1, 0.3]
    filter_by_convexity = [True, False]
    min_treshold = np.arange(2, 40, 3)
    max_treshold = [100, 175, 250]

    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    neg_elts = len([path for path in file_list if "bad" in str(path)])
    pos_elts = len([path for path in file_list if "bad" not in str(path)])

    # Put all the variables into a list, then use itertools to get all the possible combinations
    mp_args = list(product((file_list,), filter_by_area, min_area, filter_by_circularity, min_circularity,
                           filter_by_convexity, min_treshold, max_treshold))
    print(f"Starting grid search over the {len(mp_args)} possibility and {len(file_list)} images.")
    results = []
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for result in tqdm(pool.imap(grid_search_worker, mp_args, chunksize=10), total=len(mp_args)):
            results.append(result)

    stats_list = []
    for true_negs, true_pos in results:
        precision = true_pos / (true_pos + (neg_elts-true_negs))
        recall = true_pos / pos_elts
        acc = (true_pos + true_negs) / (neg_elts + pos_elts)
        pos_acc = true_pos / pos_elts
        neg_acc = true_negs / neg_elts
        stats_list.append((precision, recall, acc, pos_acc, neg_acc))

    stats = np.asarray(stats_list)
    stats_name = ("precision", "recall", "accuracy", "positive accuracy", "negative accuracy")
    best_results_idx = []
    for stat_idx in range(len(stats)):
        best_results_idx.append(np.argsort(stats, axis=stat_idx)[:5])

    print("\nFinished runnig grid_search on the dataset")
    print(f"Dataset was composed of {pos_elts} good samples and {neg_elts} bad samples")
    for stat_idx in range(len(stats)):
        print(f"Best {stats_name} results are:\n{stats[best_results_idx[stat_idx]]}"
              f"\nObtained with:\n{mp_args[best_results_idx[stat_idx]][1:]}")


def show_image(img, title: str = "Image"):
    while True:
        cv2.imshow(title, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    parser = ArgumentParser("Cuts images and corresponding masks into small tiles")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--show_missed", "--s", action="store_true",
                        help="Show samples where the blob detection failed")
    parser.add_argument("--debug", "--d", action="store_true", help="Show every sample")
    parser.add_argument("--grid_search", "--g", action="store_true", help="Use grid search to find the best parameters")
    args = parser.parse_args()

    data_path: Path = args.data_path

    if args.grid_search:
        grid_search(args.data_path)
        return

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
        if "bad" in str(img_path):
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
