#!/bin/python3

import os
import torch
import shutil
import pathlib
import numpy as np

# is needed for torch.use_deterministic_algorithms(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

np.random.seed(11)
torch.manual_seed(11)
# only works with newer PyTorch versions
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)

DTYPE_NUMPY = np.float64
DTYPE_TORCH = torch.float64

# this will only ever be set True in IK_3d_three_linkage
IS_UR5_REMOVE_CYLINDER = False

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


def compute_sample_helper(rng, limits, n):

    x = []
    for i in range(n):
        x += [limits[i][0] + rng.uniform(0, 1)*(limits[i][1] - limits[i][0])]

    return x


def compute_sample(rng, limits, is_sample_circle, radius_outer, radius_inner):

    n = len(limits)

    x = compute_sample_helper(rng, limits, n)

    if is_sample_circle:

        if radius_outer <= radius_inner:

            print(f"Make sure radius_outer > radius_inner!")
            exit(1)

        if IS_UR5_REMOVE_CYLINDER :

            assert len(x) == 3, "len(x) must be 3, if IS_UR5_REMOVE_CYLINDER == True"

        r = np.linalg.norm(x, ord=2)
        r_cyl = np.linalg.norm(x[:-1], ord=2)

        while ( r >= radius_outer or r < radius_inner ) or ( IS_UR5_REMOVE_CYLINDER and r_cyl < 0.1 ):

            x = compute_sample_helper(rng, limits, n)

            r = np.linalg.norm(x, ord=2)
            r_cyl = np.linalg.norm(x[:-1], ord=2)

    return x


def sample_joint_angles(rng, constraints):

    n = len(constraints)
    x = []

    for i in range(n):
        x += [constraints[i][0] +
              rng.uniform(0, 1)*(constraints[i][1] - constraints[i][0])]

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


def save_figure(figure, dpi, dir_path_img, fname_img):

    figure.savefig(
        fname=pathlib.Path(dir_path_img, fname_img),
        bbox_inches="tight",
        dpi=dpi
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


def compute_and_save_robot_plot(randrange, compute_energy, visualize_trajectory_and_save_image, model, x_state, is_constrained, fname, dir_path):

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

        index_batch_random = randrange(0, n_batch, 1)

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


def plot_histogram(plt, ax, arr):

    hist, bins = np.histogram(arr, bins = HIST_BINS)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(x = arr, bins = logbins, density = False, log = True, cumulative = False)

    plt.xscale('log')
    plt.grid(True)

