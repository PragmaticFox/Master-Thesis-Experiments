#!/bin/python3

# local imports
import helper

# other imports
import torch

dir_model = globals()["exp_DIR_MODEL"]
dir_path_img = globals()["exp_DIR_PATH_IMG"]

model = helper.load_model(dir_model)

helper.compute_and_save_jacobian_histogram(
    model,
    X_samples,
    dpi,
    dir_path_img,
    fname_img,
    fontdict,
    title_string
)

helper.compute_and_save_terminal_energy_histogram(
    compute_energy,
    model,
    X_samples,
    dpi,
    is_constrained,
    dir_path_img,
    fname_img,
    fontdict,
    title_string
)