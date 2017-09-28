import numpy as np
import scipy as sp
import scipy.signal
import cv2

def generatingKernel(a):

    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):

    # Make sure input is float64
    image = image.astype("float64")

    # Convolve image with kernel
    output = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)

    # Resize Output
    image_height = float(np.size(image, 1))
    image_width = float(np.size(image, 0))
    output_height = np.ceil(image_height / 2)
    output_width = np.ceil(image_width / 2)
    output = cv2.resize(output, (int(output_height), int(output_width)), interpolation=cv2.INTER_AREA)

    return output

def expand_layer(image, kernel=generatingKernel(0.4)):

    # Upsample Image
    # -----------------------------------------------------------------

    # Create 'zeroed' array twice the size of image
    (img_height, img_width) = image.shape[:2]
    output  = np.zeros([img_height * 2, img_width * 2], dtype='float64')

    # Place data from original image in output array at every other row
    output[::2,::2] = image

    # Convolve image with kernel
    output = cv2.filter2D(output, -1, kernel, borderType=cv2.BORDER_REFLECT101)
    output = output * 4

    return output

def gaussPyramid(image, levels):

    image_float_64 = image.astype("float64")

    output = [0]
    output[0] = image_float_64

    # Check to make sure levels are more than 0, return only original image otherwise
    if levels == 0:
        return output

    count = 1
    for i in range(levels):
        previous_img = output[count - 1]
        current_img = reduce_layer(previous_img)
        output.append(current_img.astype("float64"))
        count = count + 1

    return output

def laplPyramid(gaussPyr):
 
    output = []
    count = 0

    for i in range(len(gaussPyr) - 1):

        exp_img = expand_layer(gaussPyr[count + 1])

        # Crop Images as needed
        if exp_img.shape[0] != gaussPyr[count].shape[0]:
            exp_img = np.delete(exp_img, (-1), axis=0)

        if exp_img.shape[1] != gaussPyr[count].shape[1]:
            exp_img = np.delete(exp_img, (-1), axis=1)

        diff_img = gaussPyr[count] - exp_img
        output.append(diff_img)
        count = count + 1

    # Append final image from gaussPyr into output list
    output.append(gaussPyr[-1])

    return output


def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):

    output = []

    for i in range(0, len(gaussPyrMask)):

        pyr1 = gaussPyrMask[i] * laplPyrWhite[i]
        pyr2 = (1 - gaussPyrMask[i]) * laplPyrBlack[i]

        output.append(pyr1 + pyr2)

    return output


def collapse(pyramid):

    for i in range(len(pyramid)):
        print pyramid[i].shape
    count = -1
    for i in range(len(pyramid)-1,0,-1):

        layer1 = expand_layer(pyramid[count])
        layer2 = pyramid[count-1]

        # Crop Images as needed
        if layer1.shape[0] != layer2.shape[0]:
            layer1 = np.delete(layer1,(-1),axis=0)
        if layer1.shape[1] != layer2.shape[1]:
            layer1 = np.delete(layer1,(-1),axis=1)

        added_layer = layer1 + layer2

        pyramid.pop(count)
        pyramid[count] = added_layer

        print count
        print len(pyramid)


    output = pyramid[0]
    return output
