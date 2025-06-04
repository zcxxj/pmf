import numpy as np
from scipy.ndimage import gaussian_filter


def render_image(model, imsize=(105, 105), ink_ncon=3, blur_sigma=0.5, epsilon=1e-4):
    """Approximate the MATLAB render_image.m function in Python."""
    canvas = np.zeros(imsize, dtype=float)
    for stk in model['strokes']:
        p = stk['pos']
        for _ in range(len(stk['invscales'])):
            x = int(np.clip(round(p[0]), 0, imsize[0] - 1))
            y = int(np.clip(round(p[1]), 0, imsize[1] - 1))
            canvas[x, y] += 1.0
    img = canvas
    # simple broadening to mimic brush stroke width
    for _ in range(ink_ncon):
        img = gaussian_filter(img, 0, mode='constant', cval=0.0)
        img = np.minimum(1.0, img)
    if blur_sigma > 0:
        img = gaussian_filter(img, blur_sigma)
        img = gaussian_filter(img, blur_sigma)
    img = np.clip(img, 0.0, 1.0)
    return (1 - epsilon) * img + epsilon * (1 - img)
