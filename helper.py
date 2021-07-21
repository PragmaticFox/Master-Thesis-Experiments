#!/bin/python3

import os
import torch
import shutil
import random
import pathlib
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# fixes a possible "Fail to allocate bitmap" issue
# https://github.com/matplotlib/mplfinance/issues/386#issuecomment-869950969
matplotlib.use("Agg")

# is needed for torch.use_deterministic_algorithms(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(151)
np.random.seed(1611)
torch.manual_seed(171)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

DTYPE_NUMPY = np.float64
DTYPE_TORCH = torch.float64

# better leave False if you want clean plots
IS_SET_AXIS_TITLE = False

# this will only ever be set True in IK_3d_three_linkage
IS_UR5_REMOVE_CYLINDER = False
# See https://www.universal-robots.com/products/ur5-robot/
# UR5 footprint is 149mm
UR5_CYLINDER_RADIUS = 0.149

SAVEFIG_DPI = 300

N_ONE_DIM = 1000

HIST_BINS = 100

TRAIN_SAMPLE_POINTS_PLOT_SIZE_2D = 2.0
TRAIN_SAMPLE_POINTS_PLOT_SIZE_3D = 3.0

ALPHA_PARAM_3D_PLOTS = -4.0

COLORBAR_ENERGY_LOWER_THRESHOLD = 1e-5
COLORBAR_ENERGY_UPPER_THRESHOLD = 1e+0

COLORBAR_JACOBIAN_LOWER_THRESHOLD = 1e-1
COLORBAR_JACOBIAN_UPPER_THRESHOLD = 1e+3

TIME_MEASURE_UPDATE = 100
TENSORBOARD_UPDATE = 100

JOINT_PLOT_NAME = "joint_plot.png"
HEATMAP_PLOT_NAME = "heatmap_plot.png"
JACOBIAN_PLOT_NAME = "jacobian_plot.png"

HEATMAP_HISTOGRAM_NAME = "heatmap_histogram.png"
JACOBIAN_HISTOGRAM_NAME = "jacobian_histogram.png"

log_file_str = "train_eval_log_file.txt"
nn_model_full_str = "nn_model_full"
nn_model_state_dict_only_str = "nn_model_state_dict_only"

plots_fontdict = {'fontsize': 15, 'fontweight': 'normal',
                  'horizontalalignment': 'center'}


''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


class Logger(object):

    '''
        Courtesy of
        https://stackoverflow.com/a/11325249
    '''

    def __init__(self, *files):

        self.files = files

    def write(self, obj):

        for f in self.files:

            f.write(obj)
            f.flush()

    def flush(self):

        for f in self.files:

            f.flush()


def initialize_directories(directories):

    # note that the order of directory creation might matter

    for dir in directories:

        if not dir.exists():

            dir.mkdir()


def save_script(directory):

    # saves a copy of the current python script into the folder
    shutil.copy(__file__, pathlib.Path(directory, os.path.basename(__file__)))


def compute_sample_helper(limits, n):

    x = []
    for i in range(n):
        x += [limits[i][0] + random.uniform(0, 1)*(limits[i][1] - limits[i][0])]

    return x


def compute_sample(limits, is_sample_circle, radius_outer, radius_inner):

    n = len(limits)

    x = compute_sample_helper(limits, n)

    if is_sample_circle:

        if radius_outer <= radius_inner:

            print(f"Make sure radius_outer > radius_inner!")
            exit(1)

        if IS_UR5_REMOVE_CYLINDER :

            assert len(x) == 3, "len(x) must be 3, if IS_UR5_REMOVE_CYLINDER == True"

        r = np.linalg.norm(x, ord=2)
        r_cyl = np.linalg.norm(x[:-1], ord=2)

        while ( r >= radius_outer or r < radius_inner ) or ( IS_UR5_REMOVE_CYLINDER and r_cyl < UR5_CYLINDER_RADIUS):

            x = compute_sample_helper(limits, n)

            r = np.linalg.norm(x, ord=2)
            r_cyl = np.linalg.norm(x[:-1], ord=2)

    return x


def sample_joint_angles(constraints):

    n = len(constraints)
    x = []

    for i in range(n):
        x += [constraints[i][0] +
              random.uniform(0, 1)*(constraints[i][1] - constraints[i][0])]

    return x


def soft_lower_bound_constraint(limit, epsilon, stiffness, x):

    x = x - limit

    xx = torch.clone(x)

    condition = xx >= epsilon
    if len(condition) > 0 :
        x[condition] = torch.zeros_like(x[condition])

    a1 = stiffness
    b1 = -0.5 * a1 * epsilon
    c1 = -1.0 / 3 * (-b1 - a1 * epsilon) * epsilon - 1.0 / \
        2 * a1 * epsilon * epsilon - b1 * epsilon

    a2 = (-b1 - a1 * epsilon) / (epsilon * epsilon)
    b2 = a1
    c2 = b1
    d2 = c1

    y = x[xx < 0.0]
    z = x[xx < epsilon]

    x[xx < epsilon] = 1.0 / 3.0 * a2 * z * \
        z * z + 0.5 * b2 * z * z + c2 * z + d2
    x[xx < 0.0] = 0.5 * a1 * y * y + b1 * y + c1

    return x


def soft_upper_bound_constraint(limit, epsilon, stiffness, x):

    x = x - limit

    xx = torch.clone(x)

    condition = xx <= -epsilon
    if len(condition) > 0 :
        x[condition] = torch.zeros_like(x[condition])

    a1 = stiffness
    b1 = 0.5*a1*epsilon
    c1 = 1./6. * a1*epsilon*epsilon

    a2 = 1./(2.*epsilon)*a1
    b2 = a1
    c2 = 0.5*a1*epsilon
    d2 = 1./6.*a1*epsilon*epsilon

    z = x[xx > -epsilon]
    y = x[xx > 0.0]

    x[xx > -epsilon] = 1.0 / 3.0 * a2 * z * \
        z * z + 0.5 * b2 * z * z + c2 * z + d2
    x[xx > 0.0] = 0.5 * a1 * y * y + b1 * y + c1

    return x


def soft_bound_constraint(lower_limit, upper_limit, eps_rel, stiffness, x):

    epsilon = (upper_limit - lower_limit) * eps_rel

    return soft_lower_bound_constraint(lower_limit, epsilon, stiffness, x) + soft_upper_bound_constraint(upper_limit, epsilon, stiffness, x)


def compute_loss(compute_energy, model, x_state, is_constrained):

    energy, constraint, terminal_position_distance, x_hat_fk_chain = compute_energy(
        model, x_state, is_constrained)

    loss = torch.mean(energy)

    terminal_position_distance_metrics = {
        'mean': torch.mean(terminal_position_distance).item(),
        'stddev': torch.std(terminal_position_distance).item(),
        'min': torch.min(terminal_position_distance).item(),
        'max': torch.max(terminal_position_distance).item(),
        'median': torch.median(terminal_position_distance).item(),
        '75percentile': torch.quantile(terminal_position_distance, q = 0.75).item(),
        '90percentile': torch.quantile(terminal_position_distance, q = 0.90).item(),
        '95percentile': torch.quantile(terminal_position_distance, q = 0.95).item(),
        '99percentile': torch.quantile(terminal_position_distance, q = 0.99).item()
    }

    return loss, terminal_position_distance_metrics


def compute_dloss_dW(model):

    dloss_dW = 0
    weight_count = 0

    for param in model.parameters():

        if not param.grad is None:

            weight_count += 1
            param_norm = param.grad.data.norm(2)
            dloss_dW = dloss_dW + param_norm.item() ** 2

    if weight_count == 0:

        weight_count = 1

        print("[Warning in function compute_dloss_dW] Weight_count is 0, perhaps a bug?")

    dloss_dW /= weight_count

    return dloss_dW


def save_figure(figure, dpi, dir_path_img, fname_img, pad_inches = 0.1):

    figure.savefig(
        fname=pathlib.Path(dir_path_img, fname_img),
        bbox_inches="tight",
        dpi=dpi,
        pad_inches = pad_inches
        #pil_kwargs = {'optimize': True, 'quality': 75}
    )


def save_model(model, iterations, string_path, string_dict_only, string_full):
    torch.save(model, pathlib.Path(string_path, string_full))
    torch.save(model.state_dict(), pathlib.Path(string_path, string_dict_only))
    print("{} Saved Current State for Evaluation.\n".format(iterations))


def convert_sampling_mode_to_string(sampling_mode):

    sampling_string = "Sampling Once"
    if sampling_mode == 1:
        sampling_string = "Resampling"
    if sampling_mode == 2:
        sampling_string = "Expansion Sampling"

    return sampling_string


def convert_constrained_boolean_to_string(is_constrained):

    constrained_string = ""
    if is_constrained:
        constrained_string = ", Constrained"

    return constrained_string


def compute_and_save_robot_plot(compute_energy, visualize_trajectory_and_save_image, model, x_state, is_constrained, fname, dir_path):

    n_batch = x_state.shape[0]

    energy, constraint, terminal_position_distance, x_hat_fk_chain = compute_energy(
        model, x_state, is_constrained)

    index_batch_worst = np.argmax(energy.detach().tolist())

    visualize_trajectory_and_save_image(
        x_state[index_batch_worst].detach().cpu(),
        x_hat_fk_chain[index_batch_worst].detach().cpu(),
        dir_path,
        fname + "_worst_iteration.jpg"
    )

    nb = 5

    for i in range(nb):

        index_batch_random = random.randrange(0, n_batch, 1)

        visualize_trajectory_and_save_image(
            x_state[index_batch_random].detach().cpu(),
            x_hat_fk_chain[index_batch_random].detach().cpu(),
            dir_path,
            fname +
            "_random_{:d}_of_{:d}.jpg".format(i+1, nb)
        )


def compute_and_save_metrics_txt(txt_dict, test_metrics, n_iterations, dir_path_txt, fname_txt):

    txt_merge = '\nTerminal Position Distance Metrics'

    txt_merge += '\n\nTest Mean ' + str(test_metrics['mean'])

    txt_merge += '\n\nTest Stddev ' + str(test_metrics['stddev'])

    txt_merge += '\n\nTest Min ' + str(test_metrics['min'])

    txt_merge += '\n\nTest Max ' + str(test_metrics['max'])

    txt_merge += '\n\nTest Median ' + str(test_metrics['median'])

    txt_merge += '\n\nTest 75percentile ' + str(test_metrics['75percentile'])

    txt_merge += '\n\nTest 90percentile ' + str(test_metrics['90percentile'])

    txt_merge += '\n\nTest 95percentile ' + str(test_metrics['95percentile'])

    txt_merge += '\n\nTest 99percentile ' + str(test_metrics['99percentile'])

    txt_merge += f"\n\nIterations (Total: {n_iterations})\n"
    txt_merge += txt_dict['iteration']

    txt_merge += "\n\nLearning Rate\n"
    txt_merge += txt_dict['lr']

    txt_merge += '\n\nVal Mean\n'
    txt_merge += txt_dict['mean']

    txt_merge += '\n\nVal Stddev\n'
    txt_merge += txt_dict['stddev']

    txt_merge += '\n\nVal Min\n'
    txt_merge += txt_dict['min']

    txt_merge += '\n\nVal Max\n'
    txt_merge += txt_dict['max']

    txt_merge += '\n\nVal Median\n'
    txt_merge += txt_dict['median']

    txt_merge += '\n\nVal 75percentile\n'
    txt_merge += txt_dict['75percentile']

    txt_merge += '\n\nVal 90percentile\n'
    txt_merge += txt_dict['90percentile']

    txt_merge += '\n\nVal 95percentile\n'
    txt_merge += txt_dict['95percentile']

    txt_merge += '\n\nVal 99percentile\n'
    txt_merge += txt_dict['99percentile']

    with open(pathlib.Path(dir_path_txt, fname_txt), "w") as text_file :

        text_file.write(txt_merge)


def plot_histogram(plt, ax, arr, lower_thresh, upper_thresh):

    # partially inspired by
    # https://towardsdatascience.com/take-your-histograms-to-the-next-level-using-matplotlib-5f093ad7b9d3

    hist, bins = np.histogram(arr, bins = HIST_BINS)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

    n, bins, _ = ax.hist(
        x = arr,
        bins = logbins,
        density = False,
        log = True,
        cumulative = False,
        lw = 1,
        ec = "cornflowerblue",
        fc = "royalblue",
        alpha = 0.5,
        range = [lower_thresh, upper_thresh]
    )

    plt.xscale('log')
    plt.grid(False)

    plt.gca().axes.get_yaxis().set_visible(False)


def compute_jacobian(model, X_samples):

    # https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/7
    # Imagine you have n_batch samples, hence your input "matrix" will be n_batch x n_dim
    # functional.jacobian will compute the gradient of each output (n_batch x n_theta) w.r.t. each input (n_batch x n_dim).
    # By simply summing up the output over the batch dimension, now, 
    # functionl.jacobian will only compute the gradient of n_theta w.r.t. each input.
    # Since every term in the sum of one particular theta is only dependent on a single input per dimension (and not n_batch many),
    # all other sum terms will cancel out, leaving us with exactly what we want; the jacobian for each batch individually of total size n_batch x n_theta x n_dim
    # while still making use of the GPU to compute the matrix, instead of (very slowly) looping through n_batche and computing it naively.
    model_sum = lambda x : torch.sum(model(x), axis = 0)
    jac = torch.autograd.functional.jacobian(model_sum, X_samples, create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

    '''
    # sanity check whether the above does what we want (it does)
    jac_check = torch.autograd.functional.jacobian(model, X_samples, create_graph = False, strict = False, vectorize = True)
    jac_check = torch.diagonal(jac_check, dim1 = 0, dim2 = 2).permute(2, 0, 1)
    print(torch.norm(jac - jac_check, p = 2))
    '''

    '''
    # very very very slow
    # (this is just pseudocode to show the slow variant, may need to adjust to make it runnable)
    jac_slow = torch.empty()
    for i in range(X_samples.shape[0]) :
        tmp = torch.autograd.functional.jacobian(model, X_samples[i:i+1], create_graph = False, strict = False, vectorize = True)
        if jac.empty() :
            jac_slow = tmp
        else :
            jac_slow = torch.cat((jac_slow, tmp), dim = 0)
    print(torch.norm(jac - jac_slow, p = 2))
    '''

    return jac


def set_axis_title(ax, title_string, fontdict, pad = 5, is_set_axis_title = IS_SET_AXIS_TITLE):

    if is_set_axis_title :

        ax.set_title(
            title_string,
            fontdict=fontdict,
            pad=pad
        )


def create_histogram_plot(arr, title_string, fontdict, xlabel, ylabel, lower_thresh, upper_thresh, dpi, dir_path_img, fname_img):

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=1, hspace=1)

    set_axis_title(ax, title_string, fontdict)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plot_histogram(plt, ax, arr, lower_thresh, upper_thresh)

    save_figure(fig, dpi, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_jacobian_histogram(model, X_samples, dpi, dir_path_img, fname_img, fontdict, title_string):

    jac = compute_jacobian(model, X_samples)

    jac_norm = torch.norm(jac, p="fro", dim=-1)
    jac_norm = np.array(jac_norm.detach().cpu().tolist())

    arr = jac_norm.flatten()

    xlabel = "Jacobian Frobenius Norm"
    ylabel = "Samples per Bin"

    create_histogram_plot(arr, title_string, fontdict, xlabel, ylabel, COLORBAR_JACOBIAN_LOWER_THRESHOLD, COLORBAR_JACOBIAN_UPPER_THRESHOLD, dpi, dir_path_img, fname_img)


def compute_and_save_terminal_energy_histogram(compute_energy, model, X_samples, dpi, is_constrained, dir_path_img, fname_img, fontdict, title_string):

    energy, constraint, terminal_position_distance, _ = compute_energy(model, X_samples, is_constrained)

    terminal_energy = np.array(terminal_position_distance.detach().cpu().tolist())

    arr = terminal_energy.flatten()

    xlabel = "Terminal Energy [m]"
    ylabel = "Samples per Bin"

    create_histogram_plot(arr, title_string, fontdict, xlabel, ylabel, COLORBAR_ENERGY_LOWER_THRESHOLD, COLORBAR_ENERGY_UPPER_THRESHOLD, dpi, dir_path_img, fname_img)

