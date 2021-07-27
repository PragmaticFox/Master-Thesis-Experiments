#!/bin/python3

# local imports
import helper
import IK_2d_two_linkage as experiment
#import IK_3d_three_linkage as experiment

# other imports
import os
import torch
import random
import numpy as np

# is needed for torch.use_deterministic_algorithms(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.version.version}")

torch.set_default_dtype(helper.DTYPE_TORCH)

dir_model = globals()["exp_DIR_MODEL"]
dir_path_img = globals()["exp_DIR_PATH_IMG"]

model = helper.load_model(dir_model)

IS_TWOLINKAGE_CONSTRAINED = False

if "exp_IS_TWOLINKAGE_CONSTRAINED" in globals() :

    IS_TWOLINKAGE_CONSTRAINED = globals()["exp_IS_TWOLINKAGE_CONSTRAINED"]

N_SAMPLES = 1000000

if True and torch.cuda.is_available():

    device = "cuda:0"
    print("CUDA is available! Computing on GPU.")

else:

    device = "cpu"
    print("CUDA is unavailable! Computing on CPU.")

X_samples = torch.tensor(
    [
        helper.compute_sample(
            experiment.LIMITS,
            experiment.SAMPLE_CIRCLE,
            experiment.RADIUS_OUTER,
            experiment.RADIUS_INNER
        ) for _ in range(N_SAMPLES)
    ],
    dtype=helper.DTYPE_TORCH
).to(device)

dpi = 300

with torch.no_grad:

    helper.compute_and_save_jacobian_histogram(
        model,
        X_samples,
        helper.dpi,
        dir_path_img,
        "fixed_" + experiment.identifier_string + "_" + helper.HEATMAP_HISTOGRAM_NAME,
        helper.plots_fontdict,
        dir_path_img
    )

    helper.compute_and_save_terminal_energy_histogram(
        experiment.compute_energy,
        model,
        X_samples,
        dpi,
        IS_TWOLINKAGE_CONSTRAINED,
        dir_path_img,
        "fixed_" + experiment.identifier_string + "_" + helper.HEATMAP_HISTOGRAM_NAME,
        helper.plots_fontdict,
        dir_path_img
    )