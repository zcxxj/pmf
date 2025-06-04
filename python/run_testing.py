import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from motorprogram import load_motorprogram_file
from modeler import Modeler
from utils import preprocess_image


def main(result_folder, test_folder):
    model_files = []
    model_dir = os.path.join(result_folder, 'models')
    for name in sorted(os.listdir(model_dir)):
        if name.endswith('.txt'):
            model_files.append(os.path.join(model_dir, name))
    models = [load_motorprogram_file(p) for p in model_files]
    for cls, model in enumerate(models):
        test_dir = os.path.join(test_folder, str(cls))
        imname = sorted(os.listdir(test_dir))[0]
        img = preprocess_image(os.path.join(test_dir, imname))
        m = Modeler(img)
        score = m.cuckoo_search(model)
        print(f'class {cls} score {score:.2f}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python run_testing.py <result_folder> <test_folder>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
