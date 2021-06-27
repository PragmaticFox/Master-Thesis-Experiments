#!/bin/python3

import os
import io
import cv2
import sys
import math
import time
import shutil
import random
import pathlib
import datetime

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

print(f"PyTorch Version: {torch.__version__}")

SAVEFIG_DPI = 300
SAVEFIG_DPI_FINAL = 600

# is needed to torch.set_deterministic(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DTYPE_NUMPY = np.float64
DTYPE_TORCH = torch.float64
torch.set_default_dtype(DTYPE_TORCH)

# 0 is sampling once N_SAMPLES_TRAIN at the beginning of training
# 1 is resampling N_SAMPLES_TRAIN after each iteration
# 2 is expansion sampling: sampling once N_SAMPLES_TRAIN, but start with 1 sample, then add more and more samples from the vicinity.
SAMPLING_MODE = 0
IS_CONSTRAINT = False

SAMPLING_STRING = "Sampling Once"
if SAMPLING_MODE == 1 : SAMPLING_STRING = "Resampling"
if SAMPLING_MODE == 2 : SAMPLING_STRING = "Expansion Sampling"

CONSTRAINED_STRING = "Not Constrained"
if IS_CONSTRAINT: CONSTRAINED_STRING = "Constrained"

NAME_HELPER = "3d_ik_"

HEATMAP_HISTOGRAM_NAME = NAME_HELPER + "heatmap_histogram.png"
JACOBIAN_HISTOGRAM_NAME = NAME_HELPER + "jacobian_histogram.png"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# only works with newer PyTorch versions
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False
#torch.autograd.set_detect_anomaly(True)

identifier_string = "Benchmark_3d_IK"
log_file_str = "train_eval_log_file.txt"
nn_model_full_str = "nn_model_full"
nn_model_state_dict_only_str = "nn_model_state_dict_only"
dir_path_id_partial = pathlib.Path("D:/trajectory_optimization/master_thesis_experiments", identifier_string)
dtstring = str(datetime.datetime.now().replace(microsecond=0))
char_replace = [' ', '-', ':']
for c in char_replace :
    dtstring = dtstring.replace(c, '_')
dir_path_id = pathlib.Path(dir_path_id_partial, identifier_string + "_" + dtstring)
dir_path_id_model = pathlib.Path(dir_path_id, "model")
dir_path_id_plots = pathlib.Path(dir_path_id, "plots")
dir_path_id_img_val = pathlib.Path(dir_path_id, "img_val")
dir_path_id_img_train = pathlib.Path(dir_path_id, "img_train")
dir_path_id_img_test = pathlib.Path(dir_path_id, "img_test")

N_SAMPLES_TRAIN = 1000
N_SAMPLES_VAL = 1000
N_SAMPLES_TEST = 25000

N_DIM_THETA = 3
N_DIM_X = 3
N_DIM_X_STATE = 1*N_DIM_X
N_TRAJOPT = 1
N_ITERATIONS = 50000

NN_DIM_IN = 1*N_DIM_X_STATE
NN_DIM_OUT = 2*N_DIM_THETA*N_TRAJOPT
NN_DIM_IN_TO_OUT = 256

FK_ORIGIN = [0.0, 0.0, 0.0]

RADIUS_INNER = 0.25
RADIUS_OUTER = 0.9

SAMPLE_CIRCLE = True

LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

LIMITS_PLOTS = LIMITS
LIMITS_PLOTS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]
#LENGTHS = N_DIM_THETA*[(3.0 + 1e-3)/N_DIM_THETA]

LR_INITIAL = 1e-2

#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99925 # for 10k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99975 # for 30k
LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99985 # for 50k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.999925 # for 100k

TIME_MEASURE_UPDATE = 100
TENSORBOARD_UPDATE = 500
PLOT_UPATE = 10*TENSORBOARD_UPDATE


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

    def flush(self) :

        for f in self.files:

            f.flush()


class Model(torch.nn.Module):

    def mish(self, x) :

        return x * torch.tanh_(torch.log(1.0 + torch.exp(x)))

    def __init__(self):

        super(Model, self).__init__()

        self.fc_start_1 = torch.nn.Linear(NN_DIM_IN, 1*NN_DIM_IN_TO_OUT)

        self.fc_middle = torch.nn.Linear(1*NN_DIM_IN_TO_OUT, 1*NN_DIM_IN_TO_OUT)

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

        x = torch.reshape(x, shape = (x.shape[0], N_DIM_THETA*N_TRAJOPT, 2))

        # theta = arctan(y/x) = arctan(1.0*sin(theta)/1.0*cos(theta))
        theta = torch.atan2(x[:, :, 0], x[:,:, 1])
    
        #theta = - (math.pi + theta) / 2.0

        #theta = self.fc_end_alt(x)
        #theta = theta % math.pi

        return theta


def fk(theta):

    device = theta.device
    n_batch_times_n_trajOpt = theta.shape[0]
    n_dim_theta = theta.shape[1]

    p = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    p_final = torch.reshape(torch.tensor([0.0, 0.0, 0.0, 1.0]), shape = (1, 1, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1).to(device)
    rt_hom = torch.reshape(torch.eye(4,4), shape = (1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)
    rt_hom_i = torch.reshape(torch.eye(4,4), shape = (1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)

    for i in range(n_dim_theta) :

        if (i % 3 == 0) :

            # rotation around x-axis (yz-plane)
            # homogeneous coordinates

            #rt_hom_i[:, i, 0, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 1, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 2, 3] = LENGTHS[i]

            rt_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])
            rt_hom_i[:, i, 1, 2] = -torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 1] = torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])

        if (i % 3 == 1) :

            # rotation around y-axis (xz-plane)
            # homogeneous coordinates

            #rt_hom_i[:, i, 0, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 1, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 2, 3] = LENGTHS[i]

            rt_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            rt_hom_i[:, i, 0, 2] = torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 0] = -torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])

        if (i % 3 == 2) :

            # rotation around z-axis (xy-plane)
            # homogeneous coordinates

            rt_hom_i[:, i, 0, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 1, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 2, 3] = LENGTHS[i]

            rt_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            rt_hom_i[:, i, 0, 1] = -torch.sin(theta[:, i])
            rt_hom_i[:, i, 1, 0] = torch.sin(theta[:, i])
            rt_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])

        rt_hom[:, i+1] = torch.matmul(rt_hom[:, i], rt_hom_i[:, i])
        p_final[:, i+1] = torch.matmul(rt_hom[:, i+1], p)

    return p_final[:, 1:, :-1]


def compute_sample():

    x = [
        LIMITS[0][0] + random.uniform(0, 1)*(LIMITS[0][1] - LIMITS[0][0]),
        LIMITS[1][0] + random.uniform(0, 1)*(LIMITS[1][1] - LIMITS[1][0]),
        LIMITS[2][0] + random.uniform(0, 1)*(LIMITS[2][1] - LIMITS[2][0])
        ]
    
    if SAMPLE_CIRCLE :

        if RADIUS_OUTER <= RADIUS_INNER :

            print(f"Make sure RADIUS_OUTER > RADIUS_INNER!")
            exit(1)

        r = np.linalg.norm(x, ord = 2)

        while r >= RADIUS_OUTER or r < RADIUS_INNER:

            x = [
                LIMITS[0][0] + random.uniform(0, 1)*(LIMITS[0][1] - LIMITS[0][0]),
                LIMITS[1][0] + random.uniform(0, 1)*(LIMITS[1][1] - LIMITS[1][0]),
                LIMITS[2][0] + random.uniform(0, 1)*(LIMITS[2][1] - LIMITS[2][0])
                ]

            r = np.linalg.norm(x, ord = 2)

    return x


def initialize_directories():

    if not dir_path_id_partial.exists() :

        dir_path_id_partial.mkdir()

    if not dir_path_id.exists() :

        dir_path_id.mkdir()

    if not dir_path_id_plots.exists() :

        dir_path_id_plots.mkdir()

    if not dir_path_id_model.exists() :

        dir_path_id_model.mkdir()

    if not dir_path_id_img_val.exists() :

        dir_path_id_img_val.mkdir()

    if not dir_path_id_img_train.exists() :

        dir_path_id_img_train.mkdir()

    if not dir_path_id_img_test.exists() :
        
        dir_path_id_img_test.mkdir()


def compute_dloss_dW(model):

    dloss_dW = 0
    weight_count = 0

    for param in model.parameters() :

        if not param.grad is None :

            weight_count += 1
            param_norm = param.grad.data.norm(2)
            dloss_dW = dloss_dW + param_norm.item() ** 2

    if weight_count == 0 :

        weight_count = 1

        print("[Warning in function compute_dloss_dW] Weight_count is 0, perhaps a bug?")

    dloss_dW /= weight_count

    return dloss_dW


def save_figure(figure, dpi, dir_path_img, fname_img):

    figure.savefig(
        fname = pathlib.Path(dir_path_img, fname_img),
        bbox_inches = "tight",
        dpi = dpi
        #pil_kwargs = {'optimize': True, 'quality': 75}
    )


def compute_and_save_samples_plot(X_state_train, X_state_val, X_state_test, dir_path_img, fname_img):

    ax = plt.axes(projection='3d')

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], X_state_train[:, 2], ms = 1.0, marker = 'o', color = 'b', ls = '')
    ax.plot(X_state_val[:, 0], X_state_val[:, 1], X_state_val[:, 2], ms = 1.0, marker = 'o', color = 'g', ls = '')
    ax.plot(X_state_test[:, 0], X_state_test[:, 1], X_state_test[:, 2], ms = 1.0, marker = 'o', color = 'r', ls = '')

    plt.gca().set_aspect('auto', adjustable='box')

    save_figure(plt.gcf(), SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close('all')


def soft_lower_bound_constraint(limit, epsilon, stiffness, x) :

    x = x - limit
    x[x >= epsilon] = 0.0

    a1 = stiffness
    b1 = -0.5 * a1 * epsilon
    c1 = -1.0 / 3 * (-b1 - a1 * epsilon) * epsilon - 1.0 / 2 * a1 * epsilon * epsilon - b1 * epsilon

    a2 = (-b1 - a1 * epsilon) / (epsilon * epsilon)
    b2 = a1
    c2 = b1
    d2 = c1

    xx = torch.clone(x)

    y = x[xx < 0.0]
    z = x[xx < epsilon]

    x[xx < epsilon] = 1.0 / 3.0 * a2 * z * z * z + 0.5 * b2 * z * z + c2 * z + d2
    x[xx < 0.0] = 0.5 * a1 * y * y + b1 * y + c1
    
    return x


def soft_upper_bound_constraint(limit, epsilon, stiffness, x) :
 
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

    x[xx > -epsilon] = 1.0 / 3.0 * a2 * z * z * z + 0.5 * b2 * z * z + c2 * z + d2
    x[xx > 0.0] = 0.5 * a1 * y * y + b1 * y + c1

    return x


def soft_bound_constraint(lower_limit, upper_limit, eps_rel, stiffness, x) :

    epsilon = (upper_limit - lower_limit) * eps_rel

    return soft_lower_bound_constraint(lower_limit, epsilon, stiffness, x) + soft_upper_bound_constraint(upper_limit, epsilon, stiffness, x)


def compute_energy(model, x_state):

    n_batch = x_state.shape[0]

    theta_hat = model(x_state)

    x_hat_fk_chain = fk(theta_hat)

    x_hat_fk_chain = torch.reshape(input = x_hat_fk_chain, shape = (n_batch, N_TRAJOPT, N_DIM_THETA, N_DIM_X_STATE))
    x_hat_fk_chain = torch.transpose(input = x_hat_fk_chain, dim0 = 1, dim1 = 2)

    x_hat_state = x_hat_fk_chain[:, -1, -1, :]
    
    terminal_position_distance = torch.norm((x_state - x_hat_state), p = 2, dim = -1)
    energy = torch.pow(terminal_position_distance, exponent = 2)
    
    constraint_bound = torch.zeros_like(energy)

    if IS_CONSTRAINT :
        constraint_bound = soft_bound_constraint(lower_limit = -math.pi, upper_limit = 0.0, eps_rel = 1e-1, stiffness = 1e-0, x = theta_hat[:, -1])
        #constraint_bound = soft_bound_constraint(lower_limit = 0.0, upper_limit = math.pi, eps_rel = 1e-1, stiffness = 1e-0, x = theta_hat[:, -1])

    energy += constraint_bound

    return energy, constraint_bound, terminal_position_distance, x_hat_fk_chain


def compute_loss(model, x_state):

    energy, constraint, terminal_position_distance, x_hat_fk_chain = compute_energy(model, x_state)

    loss = torch.mean(energy)

    metric0 = torch.mean(terminal_position_distance)
    metric1 = torch.std(terminal_position_distance)
    metric2 = torch.max(terminal_position_distance)

    return loss, [metric0, metric1, metric2]


def save_model(model, iterations, string_path, string_dict_only, string_full):
    torch.save(model, pathlib.Path(string_path, string_full))
    torch.save(model.state_dict(), pathlib.Path(string_path, string_dict_only))
    print("{} Saved Current State for Evaluation.\n".format(iterations))


def compute_and_save_jacobian_histogram(index, model, X_samples, dpi, dir_path_img, fname_img):

    n_samples = X_samples.shape[0]

    jac = torch.zeros(size = (n_samples, N_TRAJOPT*N_DIM_THETA, N_DIM_X)).to(X_samples.device)

    for i in range(n_samples) :
        jac[i] = torch.reshape(torch.autograd.functional.jacobian(model, X_samples[i:i+1], create_graph=False, strict=False), shape = (N_TRAJOPT*N_DIM_THETA, N_DIM_X))

    jac_norm = torch.norm(jac, p = "fro", dim = -1)
    jac_norm = np.array(jac_norm.detach().cpu().tolist())

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=1, hspace=1)

    ax.set_title(
        f'\nJacobian Frobenius Norm Histogram\n3D Three-Linkage Robot Inverse Kinematics\n\nIteration {index+1}, {SAMPLING_STRING}, {CONSTRAINED_STRING}\n',
        fontdict = {'fontsize': 15, 'fontweight': 'normal', 'horizontalalignment': 'center'},
        pad = 5
    )

    arr = jac_norm.flatten() if len(jac_norm.flatten()) < 1000 else jac_norm.flatten()[:1000]

    hist, bins = np.histogram(arr, bins = 25)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(x = arr, bins = logbins, density = True, log = True)
    plt.xscale('log')
    plt.grid(True)

    save_figure(fig, dpi, dir_path_img, "histogram_" + fname_img)
    save_figure(fig, dpi, "", "histogram_jacobian_plot_3d_ik.png")

    # close the plot handle
    plt.close('all')


def compute_and_save_heatmap_histogram(index, model, X_samples, metrics, dpi, dir_path_img, fname_img):

    n_samples = X_samples.shape[0]

    terminal_energy_mean = metrics[0].detach().cpu()
    terminal_energy_std = metrics[1].detach().cpu()

    terminal_energy = torch.zeros((n_samples)).to(X_samples.device)

    energy, constraint, terminal_position_distance, _ = compute_energy(model, X_samples)

    terminal_energy = np.array(terminal_position_distance.detach().cpu().tolist())

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=1, hspace=1)

    ax.set_title(
        f'\nTerminal Energy Histogram\n3D Three-Linkage Robot Inverse Kinematics\n\nIteration {index+1}, {SAMPLING_STRING}, {CONSTRAINED_STRING}\n',
        fontdict = {'fontsize': 15, 'fontweight': 'normal', 'horizontalalignment': 'center'},
        pad = 5
    )

    arr = terminal_energy.flatten()

    hist, bins = np.histogram(arr, bins = 25)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(x = arr, bins = logbins, density = True, log = True)
    plt.xscale('log')
    plt.grid(True)

    save_figure(fig, dpi, dir_path_img, "histogram_" + fname_img)
    save_figure(fig, dpi, "", "histogram_heatmap_plot_3d_ik.png")

    # close the plot handle
    plt.close('all')


''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''

initialize_directories()

# saves a copy of the current python script into the folder
shutil.copy(__file__, pathlib.Path(dir_path_id, os.path.basename(__file__)))

if True and torch.cuda.is_available() :

    device = "cuda:0" 
    print("CUDA is available! Computing on GPU.")

else :

    device = "cpu"  
    print("CUDA is unavailable! Computing on CPU.")

device = torch.device(device)

filemode_logger = "w"

if os.path.exists(pathlib.Path(dir_path_id, log_file_str)) :

    filemode_logger = "a"

file_handle_logger = open(pathlib.Path(dir_path_id, log_file_str), mode = filemode_logger)

sys_stdout_original = sys.stdout
sys.stdout = Logger(sys_stdout_original, file_handle_logger)

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LR_INITIAL)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda epoch: LR_SCHEDULER_MULTIPLICATIVE_REDUCTION)

tb_writer = SummaryWriter()

X_state_train_all = torch.tensor([compute_sample() for _ in range(N_SAMPLES_TRAIN)], dtype = DTYPE_TORCH).to(device)
X_state_val = torch.tensor([compute_sample() for _ in range(N_SAMPLES_VAL)], dtype = DTYPE_TORCH).to(device)
X_state_test = torch.tensor([compute_sample() for _ in range(N_SAMPLES_TEST)], dtype = DTYPE_TORCH).to(device)

compute_and_save_samples_plot(X_state_train_all.detach().cpu(), X_state_val.detach().cpu(), X_state_test.detach().cpu(), dir_path_id_plots, "samples_plot.jpg")

print("\nTraining Starts!\n")

time_measure = 0
cur_index = 0
diffs = []

X_state_train = 0
distances = 0
distances_indices_sorted = 0
distance_index = 0

for j in range(N_ITERATIONS) :

    tic_loop = time.perf_counter()

    cur_index += 1
    current_lr = optimizer.param_groups[0]['lr']

    if SAMPLING_MODE == 0 :

        X_state_train = X_state_train_all

    elif SAMPLING_MODE == 1 :

        X_state_train = torch.tensor([compute_sample() for _ in range(N_SAMPLES_TRAIN)], dtype = DTYPE_TORCH).to(device)

    elif SAMPLING_MODE == 2 :

        if j == 0 :
            
            index_rng = random.randrange(0, N_SAMPLES_TRAIN)
            X_state_train = X_state_train_all[index_rng:index_rng+1]

            distances = torch.norm((X_state_train_all - X_state_train[0]), p = 2, dim = -1)
            distances_indices_sorted = torch.argsort(distances, descending = False)
            distance_index = 1

        else :

            if distance_index < N_SAMPLES_TRAIN and j % 2 == 0:

                rel_index = distances_indices_sorted[distance_index]
                
                # ablation experiment, just take the next index, not the nearest sample from the first sample
                #rel_index = distance_index

                X_state_train = torch.cat((X_state_train, X_state_train_all[rel_index:rel_index+1]), dim = 0)

                distance_index += 1

    [loss_train, metrics_train] = compute_loss(model, X_state_train)

    optimizer.zero_grad()
    loss_train.backward()
    # prevent potential exploding gradients
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()
    scheduler.step()

    if cur_index % TENSORBOARD_UPDATE == 0 or j == 0 or j == N_ITERATIONS - 1 :

        loss_val = 0
        loss_test = 0
        metrics_val = []
        metrics_test = []
        dloss_train_dW = 0

        with torch.no_grad() :

            dloss_train_dW = compute_dloss_dW(model)

            [loss_val, metrics_val] = compute_loss(model, X_state_val)

            tb_writer.add_scalar('Learning Rate', current_lr, cur_index)
            tb_writer.add_scalar('Train Loss', loss_train.detach().cpu(), cur_index)
            tb_writer.add_scalar('Mean Train Terminal Position Distance [m]', metrics_train[0].detach().cpu(), cur_index)
            tb_writer.add_scalar('Stddev Train Terminal Position Distance [m]', metrics_train[1].detach().cpu(), cur_index)
            tb_writer.add_scalar('Max Train Terminal Position Distance [m]', metrics_train[2].detach().cpu(), cur_index)
            tb_writer.add_scalar('Val Loss', loss_val.detach().cpu(), cur_index)
            tb_writer.add_scalar('Mean Val Terminal Position Distance [m]', metrics_val[0].detach().cpu(), cur_index)
            tb_writer.add_scalar('Stddev Val Terminal Position Distance [m]', metrics_val[1].detach().cpu(), cur_index)
            tb_writer.add_scalar('Max Val Terminal Position Distance [m]', metrics_val[2].detach().cpu(), cur_index)
            tb_writer.add_scalar('Loss Gradient Norm', dloss_train_dW, cur_index)

            if j == N_ITERATIONS - 1 :

                [loss_test, metrics_test] = compute_loss(model, X_state_test)

                tb_writer.add_scalar('Test Loss', loss_test.detach().cpu(), cur_index)
                tb_writer.add_scalar('Mean Test Terminal Position Distance [m]', metrics_test[0].detach().cpu(), cur_index)
                tb_writer.add_scalar('Stddev Test Terminal Position Distance [m]', metrics_test[1].detach().cpu(), cur_index)
                tb_writer.add_scalar('Max Test Terminal Position Distance [m]', metrics_test[2].detach().cpu(), cur_index)

        if cur_index % PLOT_UPATE == 0 or j == 0 or j == N_ITERATIONS - 1 :

            dpi_plots = SAVEFIG_DPI_FINAL if j == N_ITERATIONS - 1 else SAVEFIG_DPI

            metrics = metrics_val
            X_samples = X_state_val
            if j == N_ITERATIONS - 1 :
                metrics = metrics_test
                X_samples = X_state_test
        
            tic = time.perf_counter()

            compute_and_save_heatmap_histogram(model, X_samples, dpi_plots, dir_path_id_plots, cur_index, HEATMAP_HISTOGRAM_NAME)

            toc = time.perf_counter()
            print(f"{toc - tic:0.2f} [s] for compute_and_save_heatmap_histogram(...)")
            tic = time.perf_counter()

            compute_and_save_jacobian_histogram(model, X_samples, dpi_plots, dir_path_id_plots, cur_index, JACOBIAN_HISTOGRAM_NAME)
            
            toc = time.perf_counter()

            print(f"{toc - tic:0.2f} [s] for compute_and_save_jacobian_histogram(...)")

    toc_loop = time.perf_counter()
    time_measure_tmp = (toc_loop - tic_loop)
    time_measure += time_measure_tmp

    if cur_index % TIME_MEASURE_UPDATE == 0 :
        print(f"{cur_index} iterations {time_measure_tmp:0.2f} [s] (total {time_measure:0.2f} [s])")

print("\nTraining Process Completed.\n")

save_model(model, cur_index, dir_path_id_model, nn_model_state_dict_only_str, nn_model_full_str)

print("\nAll Done!\n")

sys.stdout = sys_stdout_original
file_handle_logger.close()