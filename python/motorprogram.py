import numpy as np


def load_motorprogram_file(path):
    """Load a MotorProgram saved by write_motorprogram_to_file."""
    strokes = []
    with open(path, 'r') as f:
        ns = int(f.readline())
        for _ in range(ns):
            nn = int(f.readline())
            pos = [float(x) for x in f.readline().split()]
            invscales = []
            shapes = []
            for _ in range(nn):
                invscale = float(f.readline())
                shape = [float(x) for x in f.readline().split()]
                invscales.append(invscale)
                shapes.append(shape)
            strokes.append({'pos': np.array(pos, dtype=float),
                             'invscales': np.array(invscales, dtype=float),
                             'shapes': np.array(shapes, dtype=float)})
    return {'strokes': strokes}
