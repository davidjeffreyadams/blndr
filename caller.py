import cv2
import numpy as np

import os
import errno

from os import path
from glob import glob

import blending as blend

IMG_EXTENSIONS = ["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp"]
SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"


def collect_files(prefix, extension_list=IMG_EXTENSIONS):
    """Return a list of all files in a directory that match the input prefix
    with one of the allowed extensions. """
    filenames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
    return filenames


def viz_pyramid(pyramid):
    """Create a single image by vertically stacking the levels of a pyramid."""
    shape = np.atleast_3d(pyramid[0]).shape[:-1]  # need num rows & cols only
    img_stack = [cv2.resize(layer, shape[::-1],
                            interpolation=3) for layer in pyramid]
    return np.vstack(img_stack).astype(np.uint8)


def run_blend(black_image, white_image, mask):
    """Compute the blend of two images along the boundaries of the mask.

    Assume all images are float dtype, and return a float dtype.
    """

    # Automatically figure out the size; at least 16x16 at the highest level
    min_size = min(black_image.shape)
    depth = int(np.log2(min_size)) - 4

    gauss_pyr_mask = blend.gaussPyramid(mask, depth)
    gauss_pyr_black = blend.gaussPyramid(black_image, depth)
    gauss_pyr_white = blend.gaussPyramid(white_image, depth)

    lapl_pyr_black = blend.laplPyramid(gauss_pyr_black)
    lapl_pyr_white = blend.laplPyramid(gauss_pyr_white)

    outpyr = blend.blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    img = blend.collapse(outpyr)

    return (gauss_pyr_black, gauss_pyr_white, gauss_pyr_mask,
            lapl_pyr_black, lapl_pyr_white, outpyr, [img])


def main(black_image, white_image, mask, out_path):
    """Apply pyramid blending to each color channel of the input images """

    # Convert to double and normalize the images to the range [0..1]
    # to avoid arithmetic overflow issues
    b_img = np.atleast_3d(black_image).astype(np.float) / 255.
    w_img = np.atleast_3d(white_image).astype(np.float) / 255.
    m_img = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = b_img.shape[-1]

    imgs = []
    for channel in range(num_channels):
        imgs.append(run_blend(b_img[:, :, channel],
                              w_img[:, :, channel],
                              m_img[:, :, channel]))

    try:
        os.makedirs(out_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    names = ['gauss_pyr_black', 'gauss_pyr_white', 'gauss_pyr_mask',
             'lapl_pyr_black', 'lapl_pyr_white', 'outpyr', 'outimg']

    for name, img_stack in zip(names, zip(*imgs)):
        imgs = map(np.dstack, zip(*img_stack))
        stack = [cv2.normalize(img, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX)
                 for img in imgs]
        cv2.imwrite(path.join(out_path, name + '.png'), viz_pyramid(stack))


if __name__ == "__main__":
    """Apply pyramid blending to all folders below SRC_FOLDER that contain
    a black, white, and mask image.
    """
    subfolders = os.walk(SRC_FOLDER)
    subfolders.next()  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]

        print "Processing files in '" + image_dir + "' folder..."

        black_names = collect_files(os.path.join(dirpath, '*black.'))
        white_names = collect_files(os.path.join(dirpath, '*white.'))
        mask_names = collect_files(os.path.join(dirpath, '*mask.'))

        if not len(black_names) == len(white_names) == len(mask_names) == 1:
            print("    Cannot proceed. There can only be one black, white, " +
                  "and mask image in each input folder.")
            continue

        black_img = cv2.imread(black_names[0], cv2.IMREAD_COLOR)
        white_img = cv2.imread(white_names[0], cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_names[0], cv2.IMREAD_COLOR)

        output_dir = os.path.join(OUT_FOLDER, image_dir)
        main(black_img, white_img, mask_img, output_dir)
