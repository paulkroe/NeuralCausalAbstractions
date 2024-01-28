import argparse

import numpy as np
import torch as T

import src.metric.visualization as vis
from src.ds import CTF, CTFTerm
from src.datagen import SCMDataTypes as sdt
from src.metric.model_utility import load_model


parser = argparse.ArgumentParser(description="Load trained models for evaluation.")
parser.add_argument('dir', help="directory of experiment")
parser.add_argument('--img-grids', type=int, default=1, help="number of image grids to show")
parser.add_argument('--verbose', action="store_true", help="print more information")

args = parser.parse_args()

dir = args.dir
verbose = args.verbose

m, dir_params, _, v_size, v_type = load_model(d=dir, verbose=verbose)

# Show image grids
for i in range(args.img_grids):
    img_batch = m(64, evaluating=True)
    for img_var in img_batch:
        if v_type[img_var] == sdt.IMAGE:
            vis.show_image_grid(img_batch[img_var])


if dir_params["gen"] == "CelebADataGenerator":
    #test_var = 'bald'
    test_var = 'mustache'
    #test_var = 'eyeglasses'
    y1 = CTFTerm({'image'}, {}, {'image': 1})
    x1 = CTFTerm({test_var}, {}, {test_var: 1})
    x0 = CTFTerm({test_var}, {}, {test_var: -1})
    y1dox1 = CTFTerm({'image'}, {test_var: 1}, {'image': 1})
elif dir_params["gen"] == "ColorMNISTDataGenerator":
    test_var = "digit"
    test_val_1_raw = 0
    test_val_2_raw = 5

    test_val_1 = np.zeros((1, 10))
    test_val_1[0, test_val_1_raw] = 1
    test_val_2 = np.zeros((1, 10))
    test_val_2[0, test_val_2_raw] = 1

    test_val_1 = T.from_numpy(test_val_1).float()
    test_val_2 = T.from_numpy(test_val_2)

    y1 = CTFTerm({'image'}, {}, {'image': 1})
    x1 = CTFTerm({test_var}, {}, {test_var: test_val_1})
    x0 = CTFTerm({test_var}, {}, {test_var: test_val_2})
    y1dox1 = CTFTerm({'image'}, {test_var: test_val_1}, {'image': 1})
else:
    print("Invalid data generator.")
    exit()

py1givenx1 = CTF({y1}, {x1})
py1dox1 = CTF({y1dox1}, set())
py1dox1givenx0 = CTF({y1dox1}, {x0})


queries = [py1givenx1, py1dox1, py1dox1givenx0]
#queries = [py1dox1]
for query in queries:
    for i in range(args.img_grids):
        samples = m.sample_ctf(query, n=64)
        if isinstance(samples, dict):
            for img_var in samples:
                if v_type[img_var] == sdt.IMAGE:
                    vis.show_image_grid(samples[img_var], title=str(query))
