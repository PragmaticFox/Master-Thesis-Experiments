#!/bin/python3

# local imports
import helper
#import IK_2d_two_linkage as experiment
import IK_3d_three_linkage as experiment

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

NN_DIM_IN = 1*experiment.N_DIM_X_STATE
NN_DIM_OUT = 2*experiment.N_DIM_THETA*experiment.N_TRAJOPT
NN_DIM_IN_TO_OUT = 256


class Model(torch.nn.Module):

    def mish(self, x):

        return x * torch.tanh_(torch.log(1.0 + torch.exp(x)))

    def __init__(self):

        super(Model, self).__init__()

        self.fc_start_1 = torch.nn.Linear(NN_DIM_IN, 1*NN_DIM_IN_TO_OUT)

        self.fc_middle = torch.nn.Linear(
            1*NN_DIM_IN_TO_OUT, 1*NN_DIM_IN_TO_OUT)

        self.fc_end = torch.nn.Linear(NN_DIM_IN_TO_OUT, NN_DIM_OUT)
        self.fc_end_alt = torch.nn.Linear(NN_DIM_IN_TO_OUT, NN_DIM_OUT // 2)

        self.act = None

        if "exp_activation_function" in globals() :

            exp_activation_function = globals()["exp_activation_function"]

            print("Activation Function Used: ")
            print(exp_activation_function)
            print()

            if exp_activation_function == "cos" :

                self.act = torch.cos

            if exp_activation_function == "sin" :

                self.act = torch.sin

            if exp_activation_function == "mish" :

                self.act = self.mish

            if exp_activation_function == "sigmoid" :

                self.act = torch.nn.Sigmoid()

            if exp_activation_function == "tanh" :

                self.act = torch.nn.Tanh()

            if exp_activation_function == "tanhshrink" :

                self.act = torch.nn.Tanhshrink()

            if exp_activation_function == "relu" :

                self.act = torch.nn.ReLU()

            if exp_activation_function == "leakyrelu" :

                self.act = torch.nn.LeakyReLU()
        
        else :

            self.act = torch.nn.Tanhshrink()

    def forward(self, x_in):

        x = self.fc_start_1(x_in)
        x = self.act(x)

        x = self.fc_middle(x)
        x = self.act(x)

        x = self.fc_middle(x)
        x = self.act(x)

        x = self.fc_end(x)

        x = torch.reshape(
            x, shape=(x.shape[0], experiment.N_DIM_THETA*experiment.N_TRAJOPT, 2))

        # theta = arctan(y/x) = arctan(1.0*sin(theta)/1.0*cos(theta))
        theta = torch.atan2(x[:, :, 0], x[:, :, 1])

        #theta = - (math.pi + theta) / 2.0

        #theta = self.fc_end_alt(x)
        #theta = theta % math.pi

        return theta


model = helper.load_model(dir_model)
model.eval()

IS_TWOLINKAGE_CONSTRAINED = False

if "exp_IS_TWOLINKAGE_CONSTRAINED" in globals() :

    IS_TWOLINKAGE_CONSTRAINED = globals()["exp_IS_TWOLINKAGE_CONSTRAINED"]

N_SAMPLES = 100000

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

title_string = ""

helper.compute_and_save_jacobian_histogram(
    model,
    X_samples,
    helper.SAVEFIG_DPI,
    dir_path_img,
    "fixed_" + experiment.identifier_string + "_" + helper.JACOBIAN_HISTOGRAM_NAME,
    helper.plots_fontdict,
    title_string
)

helper.compute_and_save_terminal_energy_histogram(
    experiment.compute_energy,
    model,
    X_samples,
    helper.SAVEFIG_DPI,
    IS_TWOLINKAGE_CONSTRAINED,
    dir_path_img,
    "fixed_" + experiment.identifier_string + "_" + helper.HEATMAP_HISTOGRAM_NAME,
    helper.plots_fontdict,
    title_string
)