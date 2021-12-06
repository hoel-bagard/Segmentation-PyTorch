from pathlib import Path


def get_mask_path_dice(img_path: Path) -> Path:
    # return img_path.parent / Path(str(img_path.stem) + "_segDotsTopOnly.jpg")
    return img_path.parent / Path(str(img_path.stem) + "_segDots.png")


def get_mask_path_tape(img_p: Path) -> Path:
    return img_p.parent / ("_".join(str(img_p.name).split("_")[:-1]) + "_mask_" + str(img_p.name).split("_")[-1])


def default_get_mask_path(img_p: Path) -> Path:
    return img_p.parent / (img_p.stem + "_mask.png")
