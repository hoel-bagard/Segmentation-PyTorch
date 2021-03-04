from pathlib import Path


def get_mask_path_dice(img_path: Path):
    return Path(str(img_path.stem) + "_segDotsTopOnly.png")


def get_mask_path_tape(img_p: Path):
    return img_p.parent / ("_".join(str(img_p.name).split("_")[:-1]) + "_mask_" + str(img_p.name).split("_")[-1])
