#!/bin/python3

import os
import sys
import time
import shutil
import random
import pathlib
import datetime

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# local import
import helper
#import IK_2d_two_linkage as experiment
import IK_3d_three_linkage as experiment

print(f"PyTorch Version: {torch.__version__}")

# is needed to torch.set_deterministic(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.set_default_dtype(helper.DTYPE_TORCH)

IS_ONLY_PLOT_REGION = True

# 0 is sampling once N_SAMPLES_TRAIN at the beginning of training
# 1 is resampling N_SAMPLES_TRAIN after each iteration
# 2 is expansion sampling: sampling once N_SAMPLES_TRAIN, but start with 1 sample, then add more and more samples from the vicinity.
SAMPLING_MODE = 2
IS_CONSTRAINED = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# only works with newer PyTorch versions
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)

directory_path = pathlib.Path(pathlib.Path(
    __file__).parent.resolve(), "experiments")
dir_path_id_partial = pathlib.Path(
    directory_path, experiment.identifier_string)

dtstring = str(datetime.datetime.now().replace(microsecond=0))
char_replace = [' ', '-', ':']
for c in char_replace:
    dtstring = dtstring.replace(c, '_')

dir_path_id = pathlib.Path(

    dir_path_id_partial, experiment.identifier_string + "_" + dtstring)
dir_path_id_model = pathlib.Path(dir_path_id, "model")
dir_path_id_plots = pathlib.Path(dir_path_id, "plots")

# order matters for directory creation
directories = [
    directory_path,
    dir_path_id_partial,
    dir_path_id,
    dir_path_id_model,
    dir_path_id_plots
]

N_ITERATIONS = 10000

N_SAMPLES_TRAIN = 1000
N_SAMPLES_VAL = 10000
N_SAMPLES_TEST = 100000

N_SAMPLES_THETA = 1000000

NN_DIM_IN = 1*experiment.N_DIM_X_STATE
NN_DIM_OUT = 2*experiment.N_DIM_THETA*experiment.N_TRAJOPT
NN_DIM_IN_TO_OUT = 256

LR_INITIAL = 1e-2

LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99930 # for 10k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99960 # for 20k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99965 # for 25k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99975 # for 30k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99985  # for 50k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.999925 # for 100k


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


helper.initialize_directories(directories)

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

experiment.compute_and_save_joint_angles_region_plot(
    random,
    device,
    N_SAMPLES_THETA,
    helper.SAVEFIG_DPI,
    dir_path_id_plots,
    experiment.identifier_string + "joint_angles_region_plot")

if IS_ONLY_PLOT_REGION:

    exit(0)

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_INITIAL)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda=lambda epoch: LR_SCHEDULER_MULTIPLICATIVE_REDUCTION)

tb_writer = SummaryWriter()

X_state_train_all = torch.tensor([helper.compute_sample(random, experiment.LIMITS, experiment.SAMPLE_CIRCLE, experiment.RADIUS_OUTER, experiment.RADIUS_INNER) for _ in range(
    N_SAMPLES_TRAIN)], dtype=helper.DTYPE_TORCH).to(device)
X_state_val = torch.tensor([helper.compute_sample(random, experiment.LIMITS, experiment.SAMPLE_CIRCLE, experiment.RADIUS_OUTER, experiment.RADIUS_INNER) for _ in range(
    N_SAMPLES_VAL)], dtype=helper.DTYPE_TORCH).to(device)
X_state_test = torch.tensor([helper.compute_sample(random, experiment.LIMITS, experiment.SAMPLE_CIRCLE, experiment.RADIUS_OUTER, experiment.RADIUS_INNER) for _ in range(
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

        X_state_train = torch.tensor([helper.compute_sample(random, experiment.LIMITS, experiment.SAMPLE_CIRCLE, experiment.RADIUS_OUTER, experiment.RADIUS_INNER) for _ in range(
            N_SAMPLES_TRAIN)], dtype=helper.DTYPE_TORCH).to(device)

    elif SAMPLING_MODE == 2:

        if j == 0:

            index_rng = random.randrange(0, N_SAMPLES_TRAIN)
            X_state_train = X_state_train_all[index_rng:index_rng+1]

            distances = torch.norm(
                (X_state_train_all - X_state_train[0]), p=2, dim=-1)
            distances_indices_sorted = torch.argsort(
                distances, descending=False)
            X_state_train_all = X_state_train_all[distances_indices_sorted]
            distance_index = 1

        else:

            if distance_index < N_SAMPLES_TRAIN and j % 2 == 0:

                #rel_index = distances_indices_sorted[distance_index]

                # ablation experiment, just take the next index, not the nearest sample from the first sample
                #rel_index = distance_index

                #offset = max(N_SAMPLES_TRAIN // 500, 1)
                offset = 1
                if distance_index + offset > N_SAMPLES_TRAIN :
                    offset = N_SAMPLES_TRAIN - distance_index

                X_state_train = torch.cat(
                    (X_state_train, X_state_train_all[distance_index:distance_index+offset]), dim=0)

                distance_index += offset

    [loss_train, metrics_train] = helper.compute_loss(
        experiment.compute_energy, model, X_state_train, IS_CONSTRAINED)

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

            dloss_train_dW = helper.compute_dloss_dW(model)

            [loss_val, metrics_val] = helper.compute_loss(
                experiment.compute_energy, model, X_state_val, IS_CONSTRAINED)

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

                [loss_test, metrics_test] = helper.compute_loss(
                    experiment.compute_energy, model, X_state_test, IS_CONSTRAINED)

                tb_writer.add_scalar(
                    'Test Loss', loss_test.detach().cpu(), cur_index)
                tb_writer.add_scalar(
                    'Mean Test Terminal Position Distance [m]', metrics_test[0].detach().cpu(), cur_index)
                tb_writer.add_scalar(
                    'Stddev Test Terminal Position Distance [m]', metrics_test[1].detach().cpu(), cur_index)
                tb_writer.add_scalar(
                    'Max Test Terminal Position Distance [m]', metrics_test[2].detach().cpu(), cur_index)

        if cur_index % helper.PLOT_UPATE == 0 or j == 0 or j == N_ITERATIONS - 1:

            n_one_dim = 500 if j == N_ITERATIONS - 1 else 50
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
  
            helper.compute_and_save_robot_plot(
                random.randrange,
                experiment.compute_energy,
                experiment.visualize_trajectory_and_save_image,
                model,
                X_samples,
                IS_CONSTRAINED,
                "robot_plot",
                dir_path_id_plots
            )

            toc = time.perf_counter()
            print(f"{toc - tic:0.2f} [s] for compute_and_save_robot_plot(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_joint_angles_plot(
                random,
                model,
                device,
                X_state_train,
                plot_dpi,
                n_one_dim,
                dir_path_id_plots,
                cur_index,
                experiment.identifier_string + helper.JOINT_PLOT_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_joint_angles_plot
            )


            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_joint_angles_plot(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_heatmap_plot(
                random,
                model,
                device,
                X_state_train,
                metrics,
                plot_dpi,
                IS_CONSTRAINED,
                n_one_dim,
                dir_path_id_plots,
                cur_index,
                experiment.identifier_string + helper.HEATMAP_PLOT_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_heatmap_plot
            )

            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_heatmap_plot(...)")

            tic = time.perf_counter()


            n_one_dim_jac = 200 if j == N_ITERATIONS - 1 else 20
            experiment.compute_and_save_jacobian_plot(
                random,
                model,
                device,
                X_state_train,
                plot_dpi,
                n_one_dim_jac,
                dir_path_id_plots,
                cur_index,
                experiment.identifier_string + helper.JACOBIAN_PLOT_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_jacobian_plot
            )


            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_jacobian_plot(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_heatmap_histogram(
                random,
                model,
                X_samples,
                plot_dpi,
                IS_CONSTRAINED,
                dir_path_id_plots,
                cur_index,
                experiment.identifier_string + helper.HEATMAP_HISTOGRAM_NAME,
                helper.plots_fontdict,
                string_tmp + experiment.string_title_heatmap_histogram
            )

            toc = time.perf_counter()
            print(
                f"{toc - tic:0.2f} [s] for compute_and_save_heatmap_histogram(...)")

            tic = time.perf_counter()

            experiment.compute_and_save_jacobian_histogram(
                random,
                model,
                X_samples,
                plot_dpi,
                dir_path_id_plots,
                cur_index,
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
            f"{cur_index} iterations {current_lr} lr {time_measure_tmp:0.2f} [s] (total {time_measure:0.2f} [s])")

print("\nTraining Process Completed.\n")

helper.save_model(model, cur_index, dir_path_id_model,
                  helper.nn_model_state_dict_only_str, helper.nn_model_full_str)

print("\nAll Done!\n")

sys.stdout = sys_stdout_original
file_handle_logger.close()
