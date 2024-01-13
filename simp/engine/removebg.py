import numpy as np
from PIL import Image
from .u2netp import U2netpONNX
from PIL.Image import Image as PILImage
from typing import List, Tuple

from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion

from pathlib import Path
import os

def alpha_matting_cutout(
    img: PILImage,
    mask: PILImage,
    foreground_threshold: int,
    background_threshold: int, 
    erode_structure_size: int
) -> PILImage:
    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")
    
    img = np.asarray(img)
    mask = np.asarray(mask)

    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    structure = None

    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )
    
    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout

def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout

def putalpha_cutout(img: PILImage, mask: PILImage) -> PILImage:
    img.putalpha(mask)
    return img

def apply_background_color(img: PILImage, color: Tuple[int, int, int, int]) -> PILImage:
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (r, g, b, a))
    colored_image.paste(img, mask=img)

    return colored_image

def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0,0))
    dst.paste(img2, (0, img1.height))
    return dst

def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot

def removebg(
    img: PILImage,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    putalpha: bool = False,
    post_process: bool = False
):
    size = img.size
    session = U2netpONNX("engine/assets/u2netp.onnx")
    dict_img = session.preprocessing(img)
    masks = session.predict(dict_img, size)
    cutouts = []

    for mask in masks:
        if post_process:
            mask = Image.fromarray(session.postprocessing(np.array(mask)))
        if alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                if putalpha:
                    cutout = putalpha_cutout(img, mask)
                else:
                    cutout = naive_cutout(img, mask)
        else:
            if putalpha:
                cutout = putalpha_cutout(img, mask)
            else:
                cutout = naive_cutout(img, mask)
        cutouts.append(cutout)
    
    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    return cutout
