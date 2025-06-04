import numpy as np


def load_motorprogram_file(path):
    """Load a MotorProgram saved by ``write_motorprogram_to_file``.

    The MATLAB writer pads fields using formatted ``fprintf`` so a line may
    contain multiple numbers separated by whitespace.  To be robust against
    such formatting quirks we first read every number in the file and then
    consume them sequentially according to the expected structure.
    """

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        nums = []
        for line in f:
            # Skip commented or empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split():
                try:
                    nums.append(float(tok))
                except ValueError as exc:  # pragma: no cover - malformed token
                    raise ValueError(f"cannot parse token '{tok}' in {path}") from exc

    def next_token():
        nonlocal idx
        if idx >= len(nums):
            raise ValueError(f"file '{path}' truncated while reading parameters")
        val = nums[idx]
        idx += 1
        return val

    idx = 0
    ns = int(next_token())
    stroke_meta = []
    nsub_total = 0
    for _ in range(ns):
        nn = int(next_token())
        pos = [next_token(), next_token()]
        invscales = []
        for _ in range(nn):
            invscales.append(next_token())
        stroke_meta.append({"nn": nn, "pos": pos, "invscales": invscales})
        nsub_total += nn

    remaining = len(nums) - idx
    if nsub_total == 0:
        raise ValueError("motor program contains no sub-strokes")
    if remaining % (2 * nsub_total) != 0:
        raise ValueError(
            f"cannot determine number of control points from file; {remaining} values left for {nsub_total} sub-strokes"
        )
    ncontrol = remaining // (2 * nsub_total)
    strokes = []
    for meta in stroke_meta:
        nn = meta["nn"]
        pos = meta["pos"]
        invscales = meta["invscales"]
        shapes = []
        for _ in range(nn):
            ctrl = []
            for _ in range(ncontrol):
                ctrl.append([next_token(), next_token()])
            shapes.append(ctrl)
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
