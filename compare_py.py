import skimage.measure
from scipy import ndimage


def standard_deviation(img):
    return ndimage.standard_deviation(img)

def entropy(img):
    return skimage.measure.shannon_entropy(img)

