import os

import config
import imagecorruptions.corruptions as _corr
import numpy as np
import skimage.filters as _skf
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image

_orig_gaussian = _skf.gaussian
def _gaussian_compat(image, *args, **kwargs):
    if "multichannel" in kwargs:
        mc = kwargs.pop("multichannel")
        kwargs.setdefault("channel_axis", -1 if mc else None)
    return _orig_gaussian(image, *args, **kwargs)
_corr.gaussian = _gaussian_compat


def main():
    os.makedirs(config.OUT_DIR, exist_ok=True)
    with open(config.TEST_LIST) as f:
        image_ids = [line.split()[0] for line in f if line.strip()]

    for image_id in image_ids:
        src = os.path.join(config.IMAGES_DIR, f"{image_id}.jpg")
        if not os.path.exists(src):
            continue
        img = np.array(Image.open(src).convert("RGB"))
        for corruption in get_corruption_names():
            for severity in config.SEVERITIES:
                out = corrupt(img, corruption_name=corruption, severity=severity)
                Image.fromarray(out).save(
                    os.path.join(config.OUT_DIR, f"{image_id}_{corruption}_{severity}.jpg")
                )

    print(f"done, corrupted images in {config.OUT_DIR}")


if __name__ == "__main__":
    main()