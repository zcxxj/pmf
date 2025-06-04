import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from motorprogram import load_motorprogram_file
from modeler import Modeler
from utils import preprocess_image
import numpy as np


def main(result_folder, test_folder, limit=3):
    model_files = []
    model_dir = os.path.join(result_folder, 'models')
    for name in sorted(os.listdir(model_dir)):
        if name.endswith('.txt'):
            model_files.append(os.path.join(model_dir, name))

    models = [load_motorprogram_file(p) for p in model_files]
    num_classes = len(models)
    wrong = 0
    total = 0

    for cls in range(num_classes):
        test_dir = os.path.join(test_folder, str(cls))
        images = sorted(os.listdir(test_dir))[:limit]
        for imname in images:
            img = preprocess_image(os.path.join(test_dir, imname))
            scores = []
            for model in models:
                m = Modeler(img)
                scores.append(m.cuckoo_search(model))
            pred = int(np.argmax(scores))
            if pred != cls:
                wrong += 1
            total += 1
            print(f'image {imname} true {cls} pred {pred} scores {[f"{s:.2f}" for s in scores]}')

    acc = 1 - wrong / float(total)
    print(f'accuracy {acc:.3f} over {total} images')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python run_testing.py <result_folder> <test_folder>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
