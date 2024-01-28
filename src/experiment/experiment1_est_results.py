import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from src.metric.model_utility import parse_directory


def convert_to_arrays(err_dict):
    n_list = err_dict[next(iter(err_dict))].keys()
    n_list = sorted(n_list)
    mae_lists = dict()
    ci_lists = dict()
    for exp_name in err_dict:
        mae_lists[exp_name] = []
        ci_lists[exp_name] = []
        for n in n_list:
            mae_lists[exp_name].append(np.mean(err_dict[exp_name][n]))
            ci_lists[exp_name].append(1.96 * np.std(err_dict[exp_name][n]) / np.sqrt(len(err_dict[exp_name][n])))

        mae_lists[exp_name] = np.array(mae_lists[exp_name])
        ci_lists[exp_name] = np.array(ci_lists[exp_name])

    return mae_lists, ci_lists, n_list


def all_err_plot(fig_name, err_dict, ci_dict, n_list, colors):
    for experiment in err_dict:
        plt.plot(n_list, err_dict[experiment], color=colors[experiment])
        fill_color = (colors[experiment][0], colors[experiment][1], colors[experiment][2], 0.5)
        plt.fill_between(n_list, (err_dict[experiment] - ci_dict[experiment]),
                         (err_dict[experiment] + ci_dict[experiment]), color=fill_color)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n", fontsize=22)
    plt.ylabel("MAE", fontsize=22)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


parser = argparse.ArgumentParser(description="Estimation Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('--load-file', action="store_true",
                    help="load data from existing file")
parser.add_argument('--clean', action="store_true",
                    help="delete unfinished experiments")
args = parser.parse_args()

d = args.dir
load_file = args.load_file


folders = ["naive_nonormal", "naive", "tau"]
colors = {
    "naive_nonormal": (1.0, 0.0, 0.0),
    "naive": (1.0, 0.7, 0.0),
    "tau": (0.0, 0.0, 1.0)
}


if load_file:
    err_list = np.load("{}/err_data.npy".format(d), allow_pickle=True).item()
else:
    err_list = dict()

    os.makedirs("{}/figs".format(d), exist_ok=True)
    for folder in folders:
        print("\nScanning experiments in {}...".format(folder))
        err_list[folder] = dict()

        f_dir = "{}/{}".format(d, folder)
        for t in os.listdir(f_dir):
            if os.path.isdir("{}/{}".format(f_dir, t)):
                t_dir = "{}/{}".format(f_dir, t)
                dir_params = parse_directory(t_dir)
                with open("{}/results.json".format(t_dir), 'r') as f:
                    results = json.load(f)
                n = dir_params["n_samples"]
                abs_err = abs(results["Q_err"])

                if n not in err_list[folder]:
                    err_list[folder][n] = []
                err_list[folder][n].append(abs_err)

    np.save("{}/err_data".format(d), err_list)

mae_lists, ci_lists, n_list = convert_to_arrays(err_list)
all_err_plot("{}/figs/mae.png".format(d), mae_lists, ci_lists, n_list, colors)
