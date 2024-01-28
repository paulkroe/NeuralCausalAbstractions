import os
import argparse
import numpy as np
import torch as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.metric.model_utility import load_model
from src.ds import CTFTerm, CTF
from src.datagen.color_mnist import ColorMNISTDataGenerator


def get_image_grid(ax, batch, n_rows, n_cols):
    ax.figure(figsize=(n_rows, n_cols))
    ax.axis("off")
    grid = vutils.make_grid(batch[: n_rows * n_cols], padding=2, normalize=True).cpu()
    ax.imshow(np.transpose(grid, (1, 2, 0)))


def grid_plot(data, inner_rows, inner_cols, fig_name):
    num_rows = len(data)
    num_cols = len(data[0])

    left_space = 0.0
    right_space = 1.0
    wspace = 0.0

    pv_plots = GridSpec(num_rows, num_cols, left=left_space, right=right_space,
                        top=0.95, bottom=0.1, wspace=wspace)

    fig = plt.figure(figsize=(16, 4))
    axes = []
    for row in range(num_rows):
        for col in range(num_cols):
            axes.append(fig.add_subplot(pv_plots[row * num_cols + col]))
            ax = axes[-1]

            batch = data[row][col]

            ax.axis("off")
            grid = vutils.make_grid(batch[: inner_rows * inner_cols], padding=2, normalize=True, nrow=inner_cols).cpu()
            ax.imshow(np.transpose(grid, (1, 2, 0)))

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    fig.clf()


parser = argparse.ArgumentParser(description="Colored MNIST Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
args = parser.parse_args()

d = args.dir
model_types = ["noncausal", "naive", "representational"]

n_rows = 2
n_cols = 10
n_images = n_rows * n_cols

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
y1dox1_raw = CTFTerm({'image'}, {test_var: test_val_1_raw}, {'image': 1})

py1 = CTF({y1}, set())
py1givenx1 = CTF({y1}, {x1})
py1dox1 = CTF({y1dox1}, set())
py1dox1givenx0 = CTF({y1dox1}, {x0})
py1dox1_raw = CTF({y1dox1_raw}, set())
py1dox1givenx0_raw = CTF({y1dox1_raw}, {x0})

queries = [py1, py1givenx1, py1dox1, py1dox1givenx0]
queries_raw = [py1, py1givenx1, py1dox1_raw, py1dox1givenx0_raw]

datagen = ColorMNISTDataGenerator(32, "sampling")

samples = []

index = 0
for model in model_types:
    samples.append([])
    exp_name = os.listdir("{}/{}".format(d, model))[0]
    d_model = "{}/{}/{}".format(d, model, exp_name)
    m, _, _, _, _ = load_model(d_model, verbose=False)

    for q in queries:
        samples[index].append(m.sample_ctf(q, n=n_images)["image"])

    index += 1

samples.append([])
for q in queries_raw:
    samples[-1].append(datagen.sample_ctf(q, n=n_images)["image"])

grid_plot(samples, n_rows, n_cols, "digit_results_grid.png")
