import cv2
import numpy as np
from skimage.feature import canny
from skimage import data, io, filters
from skimage.morphology import erosion,closing,opening,black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from matplotlib import pyplot as ppl
from enhancment import *
import morphsnakes
from copy import copy
from scipy.misc import imread
import pickle

#   --------------------------------------------------

#   CLAHE Algorithm


def adaptive_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image
#   --------------------------------------------------

#   --------------------------------------------------
#   Apply canny edge detection with morphologycal operations


def make_edges(image):
    blur = cv2.medianBlur(image, 9)
    edges1 = cv2.Canny(blur, 10, 50)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges1, kernel, iterations=1)
    output = edges
    #output = remove_edges_noise(edges)
    return output
#   --------------------------------------------------


#   --------------------------------------------------
#   Remove background from CT
def crop_background(image, edge):
    output = crop_face(edge, image)
    return output
#   --------------------------------------------------

#   Resize Image with new height


def resize_image(image, new_height_value):
    output = crop_img(image, new_height=new_height_value)
    return output
#   --------------------------------------------------

#   --------------------------------------------------
#   Alogrithm to apply custom threshold in image


def make_threshold(image):
    image = copy(image)
    width = image.shape[1]
    image = decreaseBrightness(image)
    if checkBrightness(image):
        threshold_image = cv2.threshold(image, 158, 255, cv2.THRESH_BINARY)[1]
    else:
        threshold_image = image & int('10000000', 2)
        retval, threshold_image = cv2.threshold(threshold_image, 20, 255, cv2.THRESH_BINARY)

    threshold_image = filter_size(threshold_image, size=50, connectivity_val=8)
    threshold_image = np.uint8(threshold_image)
    # remove noise by local position to prevent week edges in other positions
    point_expt = int(width / 2.6) - 20
    threshold_image[0:point_expt, 0:width] = cv2.medianBlur(threshold_image[0:point_expt, 0:width], 5)
    threshold_image[0:point_expt, 0:width] = filter_size(threshold_image[0:point_expt, 0:width], size=90, connectivity_val=4)

    return threshold_image
#   --------------------------------------------------

#   --------------------------------------------------
#   apply merge between two images


def remove_checkbones(image):
    width = image.shape[1]
    height = image.shape[0]
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for c in contours:
        if is_contour_bad(c, width, height) or is_contour_bad_noise(c, width, height):
            cv2.drawContours(mask, [c], -1, 0, -1)
            # mask=cv2.erode(mask ,kernel,iterations = 1)
    # -----------------------------------------------------------------------------------------------------------------------------
    output = cv2.bitwise_and(image, image, mask=mask)
    return output
#   --------------------------------------------------

#   --------------------------------------------------
#   apply merge between two images


def make_merge(edge, thresh):
    edge = cv2.bitwise_not(edge)
    merge = edge & (~thresh)
    merge = ~merge
    return merge
#   --------------------------------------------------

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u


# ------------------------------------------------------------------------------------------------------------------
def sinus_left(x, y, img):
    # g(I)
    gI = morphsnakes.gborders(img, alpha=2000, sigma=1.67)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.1, balloon=2)
    mgac.levelset = circle_levelset(img.shape, (y, x), 20)

    # Visual evolution.
    ppl.figure()
    return morphsnakes.evolve_visual(mgac, num_iters=200, background=img)


def sinus_right(x, y, img):
    # g(I)
    gI = morphsnakes.gborders(img, alpha=2000, sigma=1.67)

    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.1, balloon=2)
    mgac.levelset = circle_levelset(img.shape, (y, x), 20)

    # Visual evolution.
    ppl.figure()
    return morphsnakes.evolve_visual(mgac, num_iters=200, background=img)

#   --------------------------------------------------
#   apply active contour in image


def active_contour(img, threshold):
    thresh2, img_2, y_axis = detect_axis_y(threshold, img)
    image, x1, x2 = detect_axis_x(threshold, y=y_axis)
    im1 = sinus_right(x1, y_axis, img)
    im2 = sinus_left(x2, y_axis, img)

    im1 = im1 * 255
    im2 = im2 * 255

    im1 = im1.astype(int)
    im2 = im2.astype(int)
    img = img.astype(int)

    return im1, im2
#   --------------------------------------------------


def extract_roi(contour, image, padding=20):
    sinus = contour.astype(np.uint8)
    contours, hierarchy = cv2.findContours(sinus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    label = image[y - padding:(y + h) + padding, x - padding:(x + w) + padding]
    return label


def diagnosis_sinus(image):
    img_rows, img_cols = 200, 200
    loaded_model = pickle.load(open("images/features.pickle1.dat", "rb"))
    img = cv2.resize(image, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, -1)
    features = getFeatures(img)
    features = np.asarray(features)
    features = features.reshape(1, -1)
    h, u = loaded_model.predict_proba(features)[0]
    return h,u








