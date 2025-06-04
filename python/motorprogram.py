import numpy as np


def load_motorprogram_file(path):
    """Load a MotorProgram saved by ``write_motorprogram_to_file``.

    The MATLAB writer pads fields using formatted ``fprintf`` so a line may
    contain multiple numbers separated by whitespace.  To be robust against
    such formatting quirks we first read every number in the file and then
    consume them sequentially according to the expected structure.
    """

    with open(path, "r") as f:
        nums = []
        for line in f:
            for tok in line.split():
                nums.append(float(tok))

    idx = 0
    ns = int(nums[idx]); idx += 1
    strokes = []
    for _ in range(ns):
        nn = int(nums[idx]); idx += 1
        pos = [nums[idx], nums[idx + 1]]; idx += 2
        invscales = []
        shapes = []
        for _ in range(nn):
            invscales.append(nums[idx]); idx += 1
            shapes.append([nums[idx], nums[idx + 1]]); idx += 2
        strokes.append({
            "pos": np.asarray(pos, dtype=float),
            "invscales": np.asarray(invscales, dtype=float),
            "shapes": np.asarray(shapes, dtype=float),
        })

    return {"strokes": strokes}


def translate_motorprogram(model, dx, dy):
    """Return a copy of the model translated by (dx, dy)."""
    new_strokes = []
    for stk in model['strokes']:
        new_strokes.append({
            'pos': stk['pos'] + np.array([dx, dy]),
            'invscales': stk['invscales'].copy(),
            'shapes': stk['shapes'].copy(),
        })
    return {'strokes': new_strokes}
