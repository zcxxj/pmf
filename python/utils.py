import numpy as np
from PIL import Image


def preprocess_image(path):
    """Load an image, resize to 65x65, pad to 105x105, and binarize."""
    img = Image.open(path).convert('L')
    img = img.resize((65, 65))
    padded = Image.new('L', (105, 105), 0)
    padded.paste(img, (20, 20))
    arr = np.array(padded)
    arr[arr < 128] = 0
    arr[arr >= 128] = 1
    return arr.astype(bool)


def image_to_cloud(img):
    """Convert a binary image to a 2D point cloud as in MATLAB."""
    inds = np.argwhere(img > 0.5)
    if len(inds) == 0:
        return np.array([[-0.5, -0.5]])
    return inds / 2.0 + 0.25
