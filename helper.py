#!/bin/python3

import numpy as np
import torch
import pathlib

DTYPE_NUMPY = np.float64
DTYPE_TORCH = torch.float64

SAVEFIG_DPI = 300
SAVEFIG_DPI_FINAL = 600

TIME_MEASURE_UPDATE = 100
TENSORBOARD_UPDATE = 500
PLOT_UPATE = 10*TENSORBOARD_UPDATE

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


def soft_lower_bound_constraint(limit, epsilon, stiffness, x):

    x = x - limit
    x[x >= epsilon] = 0.0

    a1 = stiffness
    b1 = -0.5 * a1 * epsilon
    c1 = -1.0 / 3 * (-b1 - a1 * epsilon) * epsilon - 1.0 / \
        2 * a1 * epsilon * epsilon - b1 * epsilon

    a2 = (-b1 - a1 * epsilon) / (epsilon * epsilon)
    b2 = a1
    c2 = b1
    d2 = c1

    xx = torch.clone(x)

    y = x[xx < 0.0]
    z = x[xx < epsilon]

    x[xx < epsilon] = 1.0 / 3.0 * a2 * z * \
        z * z + 0.5 * b2 * z * z + c2 * z + d2
    x[xx < 0.0] = 0.5 * a1 * y * y + b1 * y + c1

    return x


def soft_upper_bound_constraint(limit, epsilon, stiffness, x):

    x = x - limit
    x[x <= -epsilon] = 0.0

    a1 = stiffness
    b1 = 0.5*a1*epsilon
    c1 = 1./6. * a1*epsilon*epsilon

    a2 = 1./(2.*epsilon)*a1
    b2 = a1
    c2 = 0.5*a1*epsilon
    d2 = 1./6.*a1*epsilon*epsilon

    xx = torch.clone(x)

    y = x[xx > 0.0]
    z = x[xx > -epsilon]

    x[xx > -epsilon] = 1.0 / 3.0 * a2 * z * \
        z * z + 0.5 * b2 * z * z + c2 * z + d2
    x[xx > 0.0] = 0.5 * a1 * y * y + b1 * y + c1

    return x


def soft_bound_constraint(lower_limit, upper_limit, eps_rel, stiffness, x):

    epsilon = (upper_limit - lower_limit) * eps_rel

    return soft_lower_bound_constraint(lower_limit, epsilon, stiffness, x) + soft_upper_bound_constraint(upper_limit, epsilon, stiffness, x)


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

    constrained_string = "Not Constrained"
    if is_constrained:
        constrained_string = "Constrained"

    return constrained_string

