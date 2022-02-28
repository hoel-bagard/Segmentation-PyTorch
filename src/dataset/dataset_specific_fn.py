from pathlib import Path


def default_get_mask_path(img_p: Path) -> Path:
    return img_p.parent / (img_p.stem + "_mask.png")
