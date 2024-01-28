import os
import sys
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, SymmetricalLogLocator
from tensorboard.backend.event_processing import event_accumulator


def running_average(nums, horizon=10):
    new_nums = [0] * len(nums)
    new_nums[0] = nums[0]
    for i in range(1, len(nums)):
        new_nums[i] = new_nums[i - 1] + nums[i]
        if i >= horizon:
            new_nums[i] -= nums[i - horizon]

    for i in range(len(nums)):
        new_nums[i] = new_nums[i] / min(i + 1, horizon)

    return new_nums


def error_plot(fig_name, iter_list, tv_errs):
    plt.plot(iter_list, np.mean(tv_errs, axis=0), color='red')
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Total MAE")
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def gaps_plot(fig_name, iter_list, q_gaps, percentiles, zoom_bounds=None, sep_bounds=None, sep_colors=None):
    q_gap_percentiles = np.percentile(q_gaps, percentiles, axis=0)

    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
    if zoom_bounds is not None:
        plt.gca().set_ylim(zoom_bounds)
    plt.plot(iter_list, q_gap_percentiles.T)
    plt.axhline(y=0.0, color='k', linestyle='-')
    if sep_bounds is not None:
        for i, b in enumerate(sep_bounds):
            plt.axhline(y=b, color=sep_colors[i], linestyle='--')
    plt.xlabel("Training Iteration")
    plt.ylabel("Max Q - Min Q")
    plt.legend(percentiles)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def all_gaps_plot(fig_name, iter_list, gap_dict, colors, zoom_bounds=None, run_avg=None):
    for experiment in gap_dict:
        q_gaps = gap_dict[experiment]
        q_mean = np.mean(q_gaps, axis=0)
        q_ci = 1.96 * np.std(q_gaps, axis=0) / np.sqrt(len(q_gaps))
        if run_avg is not None:
            q_mean = np.array(running_average(q_mean, horizon=run_avg))
            q_ci = np.array(running_average(q_ci, horizon=run_avg))
        plt.plot(iter_list, q_mean, color=colors[experiment])
        fill_color = (colors[experiment][0], colors[experiment][1], colors[experiment][2], 0.5)
        plt.fill_between(iter_list, (q_mean - q_ci), (q_mean + q_ci), color=fill_color)

    if zoom_bounds is not None:
        plt.gca().set_ylim(zoom_bounds)
    else:
        plt.gca().set_ylim([0.0, 1.0])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("Training Iteration", fontsize=22)
    plt.ylabel("Max Q - Min Q", fontsize=22)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def id_acc_plot(fig_name, iter_list, gaps_ucb_list, boundaries, run_avg=None):
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(boundaries)))))
    plt.gca().set_ylim([0.0, 1.01])
    for b in boundaries:
        gaps_ucb_sep = []
        if isinstance(gaps_ucb_list, dict):
            for exp_type in gaps_ucb_list:
                gaps = gaps_ucb_list[exp_type]
                result = (gaps <= b).astype(int)
                gaps_ucb_sep.append(result)
            gaps_ucb_sep = np.concatenate(gaps_ucb_sep, axis=0)
        else:
            gaps_ucb_sep = (gaps_ucb_list <= b).astype(int)
        acc_list = np.mean(gaps_ucb_sep, axis=0)

        if run_avg is not None:
            acc_list = running_average(acc_list, horizon=run_avg)

        plt.plot(iter_list, acc_list)
    plt.xlabel("Training Iteration")
    plt.ylabel("Correct ID %")
    plt.legend(boundaries)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


parser = argparse.ArgumentParser(description="ID Experiment Results Parser")
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


iter_skip = 10
boundaries = [0.01, 0.03, 0.05]
b_colors = ['r', 'g', 'b']
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]


if load_file:
    iter_list = np.load("{}/iters.npy".format(d), allow_pickle=True)
    all_q_gaps = np.load("{}/gaps.npy".format(d), allow_pickle=True).item()
    all_gap_ucbs = np.load("{}/gap_ucbs.npy".format(d), allow_pickle=True).item()
else:
    iters_counted = False
    iter_list = []
    all_q_gaps = dict()
    all_gap_ucbs = dict()

    os.makedirs("{}/figs".format(d), exist_ok=True)
    for folder in folders:
        print("\nScanning experiments in {}...".format(folder))

        gaps_ucb_list = []
        graph_q_gaps = []
        f_dir = "{}/{}".format(d, folder)
        for t in os.listdir(f_dir):
            if os.path.isdir("{}/{}".format(f_dir, t)):
                t_dir = "{}/{}".format(f_dir, t)
                temp_gap_list = []
                temp_tv_err_list = []
                for r in os.listdir(t_dir):
                    if os.path.isdir("{}/{}".format(t_dir, r)):
                        r_dir = "{}/{}".format(t_dir, r)
                        if os.path.isdir("{}/logs".format(r_dir)):
                            if not os.path.exists("{}/results.json".format(r_dir)):
                                if args.clean:
                                    print("Trial {}, run {} is incomplete. Deleting contents...".format(t, r))
                                    shutil.rmtree("{}".format(r_dir))
                                    if os.path.exists("{}/lock".format(t_dir)):
                                        os.remove("{}/lock".format(t_dir))
                                    if os.path.exists("{}/best.th".format(t_dir)):
                                        os.remove("{}/best.th".format(t_dir))
                                else:
                                    print("Trial {}, run {} is incomplete.".format(t, r))
                            else:
                                max_dir = "{}/logs/ncm_max/lightning_logs/version_0".format(r_dir)
                                min_dir = "{}/logs/ncm_min/lightning_logs/version_0".format(r_dir)

                                if os.path.isdir(min_dir) and os.path.isdir(max_dir):
                                    min_event = None
                                    max_event = None
                                    for item in os.listdir(min_dir):
                                        if min_event is None and "events" in item:
                                            min_event = item
                                    for item in os.listdir(max_dir):
                                        if max_event is None and "events" in item:
                                            max_event = item
                                    ea_min = event_accumulator.EventAccumulator("{}/{}".format(min_dir, min_event))
                                    ea_max = event_accumulator.EventAccumulator("{}/{}".format(max_dir, max_event))
                                    ea_min.Reload()
                                    ea_max.Reload()
                                    min_q_events = ea_min.Scalars("q_estimate")
                                    max_q_events = ea_max.Scalars("q_estimate")

                                    try:
                                        min_max_gaps = []
                                        for i in range(len(min_q_events)):
                                            if i % iter_skip == 0:
                                                iter = min_q_events[i].step
                                                min_q = np.nan_to_num(min_q_events[i].value, nan=-1.0)
                                                max_q = np.nan_to_num(max_q_events[i].value, nan=1.0)

                                                if not iters_counted:
                                                    iter_list.append(iter + 1)
                                                min_max_gaps.append(max_q - min_q)

                                        temp_gap_list.append(min_max_gaps)
                                        iters_counted = True
                                    except Exception as e:
                                        print("Error in trial {}, run {}.".format(t, r))
                                        print(e)

                if len(temp_gap_list) > 0:
                    temp_gap_list = np.array(temp_gap_list)
                    gaps_means = np.mean(temp_gap_list, axis=0)
                    graph_q_gaps.append(gaps_means)

                    if len(temp_gap_list) > 1:
                        gaps_stderr = np.std(temp_gap_list, axis=0) / np.sqrt(len(temp_gap_list))
                        gaps_ucb = gaps_means + 1.65 * gaps_stderr
                        gaps_ucb_list.append(gaps_ucb)
                        if folder not in all_gap_ucbs:
                            all_gap_ucbs[folder] = []
                        all_gap_ucbs[folder].append(gaps_ucb)

        all_q_gaps[folder] = graph_q_gaps

        # Plot gaps per graph
        gaps_plot("{}/figs/{}_gap_percentiles.png".format(d, folder), iter_list, graph_q_gaps, percentiles)

        # Plot accuracy per graph
        if len(gaps_ucb_list) > 0:
            gaps_ucb_list = np.array(gaps_ucb_list)
            id_acc_plot("{}/figs/{}_ID_classification.png".format(d, folder), iter_list, gaps_ucb_list,
                        boundaries, run_avg=None)
            id_acc_plot("{}/figs/{}_ID_classification_10runavg.png".format(d, folder), iter_list, gaps_ucb_list,
                        boundaries, run_avg=10)
            all_gap_ucbs[folder] = np.array(all_gap_ucbs[folder])

    iter_list = [iter_skip * i for i in range(len(iter_list))]

    np.save("{}/iters".format(d), iter_list)
    np.save("{}/gaps".format(d), all_q_gaps)
    np.save("{}/gap_ucbs".format(d), all_gap_ucbs)

all_gaps_plot("{}/figs/gaps_plot".format(d), iter_list, all_gap_ucbs, colors, run_avg=None)
