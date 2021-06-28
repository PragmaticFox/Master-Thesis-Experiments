#!/bin/python3

import os
import sys
import math
import time
import shutil
import random
import pathlib
import datetime

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

#import IK_2d_two_linkage as experiment
import IK_3d_three_linkage as experiment

import helper

print(f"PyTorch Version: {torch.__version__}")

# is needed to torch.set_deterministic(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.set_default_dtype(helper.DTYPE_TORCH)

# 0 is sampling once N_SAMPLES_TRAIN at the beginning of training
# 1 is resampling N_SAMPLES_TRAIN after each iteration
# 2 is expansion sampling: sampling once N_SAMPLES_TRAIN, but start with 1 sample, then add more and more samples from the vicinity.
SAMPLING_MODE = 0
IS_CONSTRAINED = True

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# only works with newer PyTorch versions
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.benchmark = False
#torch.autograd.set_detect_anomaly(True)

#directory_path = "D:/trajectory_optimization/master_thesis_experiments"
directory_path = pathlib.Path(__file__).parent.resolve()
dir_path_id_partial = pathlib.Path(directory_path, "experiments/" + experiment.identifier_string)

dtstring = str(datetime.datetime.now().replace(microsecond=0))
char_replace = [' ', '-', ':']
for c in char_replace:
    dtstring = dtstring.replace(c, '_')

dir_path_id = pathlib.Path(
    dir_path_id_partial, experiment.identifier_string + "_" + dtstring)
dir_path_id_model = pathlib.Path(dir_path_id, "model")
dir_path_id_plots = pathlib.Path(dir_path_id, "plots")

N_ITERATIONS = 50000

N_SAMPLES_TRAIN = 1000
N_SAMPLES_VAL = 1000
N_SAMPLES_TEST = 10000

NN_DIM_IN = 1*experiment.N_DIM_X_STATE
NN_DIM_OUT = 2*experiment.N_DIM_THETA*experiment.N_TRAJOPT
NN_DIM_IN_TO_OUT = 256


''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


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

        #self.act = torch.cos
        #self.act = torch.sin
        #self.act = self.mish
        #self.act = torch.nn.Identity()
        #self.act = torch.nn.Sigmoid()
        #self.act = torch.nn.Tanh()
        #self.act = torch.nn.ReLU()
        #self.act = torch.nn.GELU()
        #self.act = torch.nn.Softplus()
        #self.act = torch.nn.LogSigmoid()
        #self.act = torch.nn.ELU()
        #self.act = torch.nn.SELU()
        #self.act = torch.nn.LeakyReLU()
        #self.act = torch.nn.PReLU()
        #self.act = torch.nn.SiLU()
        #self.act = torch.nn.RReLU()
        self.act = torch.nn.Tanhshrink()
        #self.act = torch.nn.Hardshrink()
        #self.act = torch.nn.Hardtanh()
        #self.act = torch.nn.Hardswish()
        #self.act = torch.nn.ReLU6()
        #self.act = torch.nn.CELU()
        #self.act = torch.nn.Softshrink()
        #self.act = torch.nn.Softsign()

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


def initialize_directories():

    if not dir_path_id_partial.exists():

        dir_path_id_partial.mkdir()

    if not dir_path_id.exists():

        dir_path_id.mkdir()

    if not dir_path_id_plots.exists():

        dir_path_id_plots.mkdir()

    if not dir_path_id_model.exists():

        dir_path_id_model.mkdir()


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


def compute_loss(model, x_state):

    energy, constraint, terminal_position_distance, x_hat_fk_chain = experiment.compute_energy(
        model, x_state, IS_CONSTRAINED)

    loss = torch.mean(energy)

    metric0 = torch.mean(terminal_position_distance)
    metric1 = torch.std(terminal_position_distance)
    metric2 = torch.max(terminal_position_distance)

    return loss, [metric0, metric1, metric2]


def save_model(model, iterations, string_path, string_dict_only, string_full):
    torch.save(model, pathlib.Path(string_path, string_full))
    torch.save(model.state_dict(), pathlib.Path(string_path, string_dict_only))
    print("{} Saved Current State for Evaluation.\n".format(iterations))


def compute_and_save_robot_plot(model, x_state, index, fname, dir_path):

    n_batch = x_state.shape[0]

    energy, constraint, terminal_position_distance, x_hat_fk_chain = experiment.compute_energy(
        model, x_state, IS_CONSTRAINED)

    index_batch_worst = np.argmax(energy.detach().tolist())

    #print(x_hat_fk_chain[index_batch_worst].shape)

    experiment.visualize_trajectory_and_save_image(
        x_state[index_batch_worst].detach().cpu(),
        x_hat_fk_chain[index_batch_worst].detach().cpu(),
        dir_path,
        fname + "_worst_iteration_{:d}.jpg".format(index+1)
    )

    nb = 10

    for i in range(nb):

        index_batch_random = random.randrange(0, n_batch, 1)

        experiment.visualize_trajectory_and_save_image(
            x_state[index_batch_random].detach().cpu(),
            x_hat_fk_chain[index_batch_random].detach().cpu(),
            dir_path,
            fname +
            "_random_{:d}_of_{:d}_iteration_{:d}.jpg".format(i+1, nb, index+1)
        )


''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


initialize_directories()

# saves a copy of the current python script into the folder
shutil.copy(__file__, pathlib.Path(dir_path_id, os.path.basename(__file__)))

if True and torch.cuda.is_available():

    device = "cuda:0"
    print("CUDA is available! Computing on GPU.")

else:

    device = "cpu"
    print("CUDA is unavailable! Computing on CPU.")

device = torch.device(device)

filemode_logger = "w"

if os.path.exists(pathlib.Path(dir_path_id, helper.log_file_str)):

    filemode_logger = "a"

file_handle_logger = open(pathlib.Path(
    dir_path_id, helper.log_file_str), mode=filemode_logger)

sys_stdout_original = sys.stdout
sys.stdout = helper.Logger(sys_stdout_original, file_handle_logger)

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=experiment.LR_INITIAL)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda=lambda epoch: experiment.LR_SCHEDULER_MULTIPLICATIVE_REDUCTION)

tb_writer = SummaryWriter()

X_state_train_all = torch.tensor([experiment.compute_sample() for _ in range(
    N_SAMPLES_TRAIN)], dtype=helper.DTYPE_TORCH).to(device)
X_state_val = torch.tensor([experiment.compute_sample() for _ in range(
    N_SAMPLES_VAL)], dtype=helper.DTYPE_TORCH).to(device)
X_state_test = torch.tensor([experiment.compute_sample() for _ in range(
    N_SAMPLES_TEST)], dtype=helper.DTYPE_TORCH).to(device)

experiment.compute_and_save_samples_plot(X_state_train_all.detach().cpu(), X_state_val.detach(
).cpu(), X_state_test.detach().cpu(), dir_path_id_plots, "samples_plot.jpg")

print("\nTraining Starts!\n")

time_measure = 0
cur_index = 0
diffs = []

X_state_train = 0

distances = 0
distance_index = 0
distances_indices_sorted = 0

for j in range(N_ITERATIONS):

    tic_loop = time.perf_counter()

    cur_index += 1
    current_lr = optimizer.param_groups[0]['lr']

    if SAMPLING_MODE == 0:

        X_state_train = X_state_train_all

    elif SAMPLING_MODE == 1:

        X_state_train = torch.tensor([experiment.compute_sample() for _ in range(
            N_SAMPLES_TRAIN)], dtype=helper.DTYPE_TORCH).to(device)

    elif SAMPLING_MODE == 2:

        if j == 0:

            index_rng = random.randrange(0, N_SAMPLES_TRAIN)
            X_state_train = X_state_train_all[index_rng:index_rng+1]

            distances = torch.norm(
                (X_state_train_all - X_state_train[0]), p=2, dim=-1)
            distances_indices_sorted = torch.argsort(
                distances, descending=False)
            distance_index = 1

        else:

            if distance_index < N_SAMPLES_TRAIN and j % 2 == 0:

                rel_index = distances_indices_sorted[distance_index]

                # ablation experiment, just take the next index, not the nearest sample from the first sample
                #rel_index = distance_index

                X_state_train = torch.cat(
                    (X_state_train, X_state_train_all[rel_index:rel_index+1]), dim=0)

                distance_index += 1

    [loss_train, metrics_train] = compute_loss(model, X_state_train)

    optimizer.zero_grad()
    loss_train.backward()
    # prevent potential exploding gradients
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()
    scheduler.step()

    if cur_index % helper.TENSORBOARD_UPDATE == 0 or j == 0 or j == N_ITERATIONS - 1:

        loss_val = 0
        loss_test = 0
        metrics_val = []
        metrics_test = []
        dloss_train_dW = 0

        with torch.no_grad():

            dloss_train_dW = compute_dloss_dW(model)

            [loss_val, metrics_val] = compute_loss(model, X_state_val)

            tb_writer.add_scalar('Learning Rate', current_lr, cur_index)
            tb_writer.add_scalar(
                'Train Loss', loss_train.detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Mean Train Terminal Position Distance [m]', metrics_train[0].detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Stddev Train Terminal Position Distance [m]', metrics_train[1].detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Max Train Terminal Position Distance [m]', metrics_train[2].detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Val Loss', loss_val.detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Mean Val Terminal Position Distance [m]', metrics_val[0].detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Stddev Val Terminal Position Distance [m]', metrics_val[1].detach().cpu(), cur_index)
            tb_writer.add_scalar(
                'Max Val Terminal Position Distance [m]', metrics_val[2].detach().cpu(), cur_index)
            tb_writer.add_scalar('Loss Gradient Norm',
                                 dloss_train_dW, cur_index)

            if j == N_ITERATIONS - 1:

                [loss_test, metrics_test] = compute_loss(model, X_state_test)

                tb_writer.add_scalar(
                    'Test Loss', loss_test.detach().cpu(), cur_index)
                tb_writer.add_scalar(
                    'Mean Test Terminal Position Distance [m]', metrics_test[0].detach().cpu(), cur_index)
                tb_writer.add_scalar(
                    'Stddev Test Terminal Position Distance [m]', metrics_test[1].detach().cpu(), cur_index)
                tb_writer.add_scalar(
                    'Max Test Terminal Position Distance [m]', metrics_test[2].detach().cpu(), cur_index)

        if cur_index % helper.PLOT_UPATE == 0 or j == 0 or j == N_ITERATIONS - 1:

            n_one_dim = 1000 if j == N_ITERATIONS - 1 else 50
            plot_dpi = helper.SAVEFIG_DPI_FINAL if j == N_ITERATIONS - 1 else helper.SAVEFIG_DPI

            metrics = metrics_val
            X_samples = X_state_val

            if j == N_ITERATIONS - 1:
                metrics = metrics_test
                X_samples = X_state_test

            constrained_string = helper.convert_constrained_boolean_to_string(
                IS_CONSTRAINED)
            sampling_string = helper.convert_sampling_mode_to_string(
                SAMPLING_MODE)

            string_tmp = f'\nIteration {cur_index}, {sampling_string}, {constrained_string}\n'

            tic = time.perf_counter()

            compute_and_save_robot_plot(model, X_samples, cur_index, "robot_plot", dir_path_id_plots)

            toc = time.perf_counter()
            print(f"{toc - tic:0.2f} [s] for compute_and_save_robot_plot(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_joint_angles_plot(
                model, device, plot_dpi, n_one_dim, dir_path_id_plots, cur_index,
                experiment.identifier_string + helper.JOINT_PLOT_NAME,
                string_tmp + experiment.string_title_joint_angles_plot
            )

            toc = time.perf_counter()
            print(f"{toc - tic:0.2f} [s] for compute_and_save_joint_angles_plot(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_heatmap_plot(
                model, device, X_state_train, metrics, plot_dpi, IS_CONSTRAINED, n_one_dim, dir_path_id_plots, cur_index,
                experiment.identifier_string + helper.HEATMAP_PLOT_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_heatmap_plot
            )

            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_heatmap_plot(...)")

            tic = time.perf_counter()

            n_one_dim_jac = 300 if j == N_ITERATIONS - 1 else 30
            experiment.compute_and_save_jacobian_plot(
                model, device, X_state_train, plot_dpi, n_one_dim_jac, dir_path_id_plots, cur_index,
                experiment.identifier_string + helper.JACOBIAN_PLOT_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_jacobian_plot
            )

            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_jacobian_plot(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_heatmap_histogram(
                model, X_samples, plot_dpi, IS_CONSTRAINED, dir_path_id_plots, cur_index,
                experiment.identifier_string + helper.HEATMAP_HISTOGRAM_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_heatmap_histogram
            )

            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_heatmap_histogram(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_jacobian_histogram(
                model, X_samples, plot_dpi, dir_path_id_plots, cur_index,
                experiment.identifier_string + helper.JACOBIAN_HISTOGRAM_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_jacobian_histogram
            )

            toc = time.perf_counter()

            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_jacobian_histogram(...)")

    toc_loop = time.perf_counter()
    time_measure_tmp = (toc_loop - tic_loop)
    time_measure += time_measure_tmp

    if cur_index % helper.TIME_MEASURE_UPDATE == 0:
        print(
            f"{cur_index} iterations {time_measure_tmp:0.2f} [s] (total {time_measure:0.2f} [s])")

print("\nTraining Process Completed.\n")

save_model(model, cur_index, dir_path_id_model,
           helper.nn_model_state_dict_only_str, helper.nn_model_full_str)

print("\nAll Done!\n")

sys.stdout = sys_stdout_original
file_handle_logger.close()
