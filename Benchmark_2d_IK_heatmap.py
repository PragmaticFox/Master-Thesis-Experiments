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

SAVEFIG_DPI = 100
SAVEFIG_DPI_FINAL = 300

# is needed to torch.set_deterministic(True) below
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DTYPE_NUMPY = np.float64
DTYPE_TORCH = torch.float64
torch.set_default_dtype(DTYPE_TORCH)

# 0 is sampling once N_SAMPLES_TRAIN at the beginning of training
# 1 is resampling N_SAMPLES_TRAIN after each iteration
# 2 is sampling once N_SAMPLES_TRAIN, but start with 1 sample, then add more and more samples from the vicinity.
SAMPLING_MODE = 1
IS_CONSTRAINT = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# only works with newer PyTorch versions
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False
#torch.autograd.set_detect_anomaly(True)

identifier_string = "Benchmark_2d_IK_heatmap"
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
dir_path_id_joint_plot = pathlib.Path(dir_path_id, "joint_plot")
dir_path_id_jacobian_visualization = pathlib.Path(dir_path_id, "jacobian_visualization")
dir_path_id_heatmap = pathlib.Path(dir_path_id, "heatmap")
dir_path_id_img_val = pathlib.Path(dir_path_id, "img_val")
dir_path_id_img_train = pathlib.Path(dir_path_id, "img_train")
dir_path_id_img_test = pathlib.Path(dir_path_id, "img_test")
dir_path_id_img_samples = pathlib.Path(dir_path_id, "img_samples")

N_SAMPLES_TRAIN = 10
N_SAMPLES_VAL = 1000
N_SAMPLES_TEST = 25000

N_DIM_THETA = 2
N_DIM_X = 2
N_DIM_X_STATE = 1*N_DIM_X
N_TRAJOPT = 1
N_ITERATIONS = 50000

NN_DIM_IN = 1*N_DIM_X_STATE
NN_DIM_OUT = 2*N_DIM_THETA*N_TRAJOPT
NN_DIM_IN_TO_OUT = 256

FK_ORIGIN = [0.0, 0.0]

RADIUS_INNER = 0.25
RADIUS_OUTER = 0.9

SAMPLE_CIRCLE = True

LIMITS = [[-1.0, 1.0], [-1.0, 1.0]]

LIMITS_HEATMAP = LIMITS
LIMITS_HEATMAP = [[-1.0, 1.0], [-1.0, 1.0]]

LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]

LR_INITIAL = 1e-2

#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99925 # for 10k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99975 # for 30k
LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99985 # for 50k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.999925 # for 100k

TIME_MEASURE_UPDATE = 100
TENSORBOARD_UPDATE = 5000
FK_VISUALIZATION_UPDATE = 10000


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

    theta_accum = torch.zeros((theta.shape[0])).to(theta.device)
    p = torch.zeros((theta.shape[0], theta.shape[1] + 1, N_DIM_X)).to(device)

    for i in range(theta.shape[1]) :

        theta_accum = theta_accum + theta[:, i]

        p[:, i+1, 0] = p[:, i, 0] + float(LENGTHS[i]) * torch.cos(theta_accum)
        p[:, i+1, 1] = p[:, i, 1] + float(LENGTHS[i]) * torch.sin(theta_accum)

    return p[:, 1:, :]


def compute_sample():

    x = [LIMITS[0][0] + random.uniform(0, 1)*(LIMITS[0][1] - LIMITS[0][0]), LIMITS[1][0] + random.uniform(0, 1)*(LIMITS[1][1] - LIMITS[1][0])]
    
    if SAMPLE_CIRCLE :

        if RADIUS_OUTER <= RADIUS_INNER :

            print(f"Make sure RADIUS_OUTER > RADIUS_INNER!")
            exit(1)

        r = np.linalg.norm(x, ord = 2)

        while r >= RADIUS_OUTER or r < RADIUS_INNER:

            x = [LIMITS[0][0] + random.uniform(0, 1)*(LIMITS[0][1] - LIMITS[0][0]), LIMITS[1][0] + random.uniform(0, 1)*(LIMITS[1][1] - LIMITS[1][0])]
            r = np.linalg.norm(x, ord = 2)

    return x


def draw_fk_joint(origin, joint_pos, index, alpha):

    plt.arrow(
        origin[0],
        origin[1],
        joint_pos[0][index][0] - origin[0],
        joint_pos[0][index][1] - origin[1],
        length_includes_head = True,
        width = 0.005,
        head_length = 0.015,
        head_width = 0.025,
        fc = 'tab:blue',
        ec = 'tab:blue',
        alpha = alpha
    )

    for i in range(1, len(joint_pos), 1) :

        plt.arrow(
            joint_pos[i-1][index][0],
            joint_pos[i-1][index][1],
            joint_pos[i][index][0] - joint_pos[i-1][index][0],
            joint_pos[i][index][1] - joint_pos[i-1][index][1],
            length_includes_head = True,
            width = 0.0075,
            head_length = 0.015,
            head_width = 0.025,
            fc = 'tab:blue',
            ec = 'tab:blue',
            alpha = alpha
        )


def initialize_directories():

    if not dir_path_id_partial.exists() :

        dir_path_id_partial.mkdir()

    if not dir_path_id.exists() :

        dir_path_id.mkdir()

    if not dir_path_id_joint_plot.exists() :

        dir_path_id_joint_plot.mkdir()

    if not dir_path_id_jacobian_visualization.exists() :

        dir_path_id_jacobian_visualization.mkdir()

    if not dir_path_id_heatmap.exists() :

        dir_path_id_heatmap.mkdir()

    if not dir_path_id_model.exists() :

        dir_path_id_model.mkdir()

    if not dir_path_id_img_val.exists() :

        dir_path_id_img_val.mkdir()

    if not dir_path_id_img_train.exists() :

        dir_path_id_img_train.mkdir()

    if not dir_path_id_img_test.exists() :
        
        dir_path_id_img_test.mkdir()

    if not dir_path_id_img_samples.exists() :
        
        dir_path_id_img_samples.mkdir()


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


def visualize_trajectory_and_save_image(x_state_list, x_hat_fk_chain_list, dir_path_img, fname_img):

    plt.scatter(x_state_list[0], x_state_list[1], c = 'r', s = 100, zorder = -10)

    for t in range(N_TRAJOPT) :
        
        plt.arrow(
            FK_ORIGIN[0],
            FK_ORIGIN[1],
            x_hat_fk_chain_list[0][t][0] - FK_ORIGIN[0],
            x_hat_fk_chain_list[0][t][1] - FK_ORIGIN[1],
            length_includes_head = True,
            width = 0.005,
            head_length = 0.015,
            head_width = 0.025,
            fc = 'tab:blue',
            ec = 'tab:blue'
        )

        for i in range(1, len(x_hat_fk_chain_list), 1) :

            plt.scatter(x_hat_fk_chain_list[i-1][t][0], x_hat_fk_chain_list[i-1][t][1], c = '0.5', s = 5)

            plt.arrow(
                x_hat_fk_chain_list[i-1][t][0],
                x_hat_fk_chain_list[i-1][t][1],
                x_hat_fk_chain_list[i][t][0] - x_hat_fk_chain_list[i-1][t][0],
                x_hat_fk_chain_list[i][t][1] - x_hat_fk_chain_list[i-1][t][1],
                length_includes_head = True,
                width = 0.0075,
                head_length = 0.015,
                head_width = 0.025,
                fc = 'tab:blue',
                ec = 'tab:blue'
            )

        plt.scatter(x_hat_fk_chain_list[-1][t][0], x_hat_fk_chain_list[-1][t][1], c = 'k', s = 25)

    plt.scatter(FK_ORIGIN[0], FK_ORIGIN[1], c = '0.5', s = 5)

    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.gca().set_aspect('equal', adjustable='box')

    save_figure(plt.gcf(), SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close()


def compute_and_save_samples_plot(X_state_train, X_state_val, X_state_test, dir_path_img, fname_img):

    plt.plot(X_state_train[:, 0], X_state_train[:, 1], ms = 1.0, marker = 'o', color = 'b', ls = '')
    plt.plot(X_state_val[:, 0], X_state_val[:, 1], ms = 1.0, marker = 'o', color = 'g', ls = '')
    plt.plot(X_state_test[:, 0], X_state_test[:, 1], ms = 1.0, marker = 'o', color = 'r', ls = '')

    plt.gca().set_aspect('equal', adjustable='box')

    save_figure(plt.gcf(), SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close()


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


def compute_loss(model, x_state, iteration_index, is_visualize, fname, dir_path):

    n_batch = x_state.shape[0]

    energy, constraint, terminal_position_distance, x_hat_fk_chain = compute_energy(model, x_state)

    loss = torch.mean(energy)

    metric0 = torch.mean(terminal_position_distance)
    metric1 = torch.std(terminal_position_distance)
    metric2 = torch.max(terminal_position_distance)

    if is_visualize :

        index_batch_worst = np.argmax(energy.detach().tolist())

        visualize_trajectory_and_save_image(
            x_state[index_batch_worst].detach().tolist(),
            x_hat_fk_chain[index_batch_worst].detach().tolist(),
            dir_path,
            fname + "_worst_{:d}.jpg".format(iteration_index+1)
        )

        index_batch_random = random.randint(0, n_batch-1)

        visualize_trajectory_and_save_image(
            x_state[index_batch_random].detach().tolist(),
            x_hat_fk_chain[index_batch_random].detach().tolist(),
            dir_path,
            fname + "_random_{:d}.jpg".format(iteration_index+1)
        )

    return loss, [metric0, metric1, metric2]


def save_model(model, iterations, string_path, string_dict_only, string_full):
    torch.save(model, pathlib.Path(string_path, string_full))
    torch.save(model.state_dict(), pathlib.Path(string_path, string_dict_only))
    print("{} Saved Current State for Evaluation.\n".format(iterations))


def compute_and_save_joint_plot(model, device, dpi, n_one_direction, dir_path_img, fname_img):

    alpha = 0.5

    dimX = np.linspace(LIMITS_HEATMAP[0][0], LIMITS_HEATMAP[0][1], n_one_direction)
    dimY = np.linspace(LIMITS_HEATMAP[1][0], LIMITS_HEATMAP[1][1], n_one_direction)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(np.stack((dimX.flatten(), dimY.flatten()), axis = -1)).to(device)

    theta_hat = torch.zeros((n_one_direction*n_one_direction, N_TRAJOPT, N_DIM_THETA))

    with torch.no_grad() :

        if n_one_direction > 100 :

            n_splits = 100

            delta = n_one_direction*n_one_direction // n_splits

            for split in range(n_splits):
                theta_hat_tmp = model(x_state[split*delta:(split+1)*delta])
                #print(theta_hat_tmp.shape)
                #print(theta_hat[split*delta:(split+1)*delta].shape)
                #print(torch.reshape(theta_hat_tmp, (delta, N_TRAJOPT, N_DIM_THETA)).shape)
                theta_hat[split*delta:(split+1)*delta] = torch.reshape(theta_hat_tmp, (delta, N_TRAJOPT, N_DIM_THETA))

        else :

            theta_hat = model(x_state)

    theta_hat = ( theta_hat % ( 2.0 * math.pi ) ) * 180.0 / math.pi

    theta_hat = torch.reshape(input = theta_hat, shape = (n_one_direction, n_one_direction, N_TRAJOPT, N_DIM_THETA)).detach().cpu()

    rad_min = 0.0
    rad_max = 360.0

    theta_hat_1 = theta_hat[:, :,  -1, 0]
    theta_hat_2 = theta_hat[:, :, -1, 1]

    fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True)

    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=1, hspace=0.25)

    axes[0].set_aspect(aspect = 'equal', adjustable = 'box')
    axes[1].set_aspect(aspect = 'equal', adjustable = 'box')

    axes[0].set_title(
        '\nJoint 1 Angles [deg]\n',
        fontdict = {'fontsize': 10, 'fontweight': 'normal', 'horizontalalignment': 'center'},
        pad = 5
    )

    axes[0].axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])

    c = axes[0].pcolormesh(dimX, dimY, theta_hat_1, cmap = 'twilight', shading = 'gouraud', vmin = rad_min, vmax = rad_max)

    axes[1].set_aspect(aspect = 'equal', adjustable = 'box')

    axes[1].set_title(
        '\nJoint 2 Angles [deg]\n',
        fontdict = {'fontsize': 10, 'fontweight': 'normal', 'horizontalalignment': 'center'},
        pad = 5
    )

    axes[1].axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])

    c = axes[1].pcolormesh(dimX, dimY, theta_hat_2, cmap = 'twilight', shading = 'gouraud', vmin = rad_min, vmax = rad_max)

    cb = fig.colorbar(c, ax = axes.ravel().tolist(), extend = 'max')

    if SAMPLE_CIRCLE :
        circleInner1 = plt.Circle((0.0, 0.0), radius = RADIUS_INNER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)
        circleOuter1 = plt.Circle((0.0, 0.0), radius = RADIUS_OUTER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)
        circleInner2 = plt.Circle((0.0, 0.0), radius = RADIUS_INNER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)
        circleOuter2 = plt.Circle((0.0, 0.0), radius = RADIUS_OUTER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)

    if LIMITS_HEATMAP != LIMITS :
        rectangle1 = plt.Rectangle(xy = (LIMITS[0][0], LIMITS[1][0]), width = LIMITS[0][1]-LIMITS[0][0], height = LIMITS[1][1]-LIMITS[1][0], color = 'orange', fill = False, lw = 4.0, alpha = alpha)
        rectangle2 = plt.Rectangle(xy = (LIMITS[0][0], LIMITS[1][0]), width = LIMITS[0][1]-LIMITS[0][0], height = LIMITS[1][1]-LIMITS[1][0], color = 'orange', fill = False, lw = 4.0, alpha = alpha)
    
    save_figure(fig, dpi, dir_path_img, "no_train_region_" + fname_img)

    if LIMITS_HEATMAP != LIMITS or SAMPLE_CIRCLE:

        if SAMPLE_CIRCLE :
            axes[0].add_patch(circleInner1)
            axes[0].add_patch(circleOuter1)
            axes[1].add_patch(circleInner2)
            axes[1].add_patch(circleOuter2)

        if LIMITS_HEATMAP != LIMITS:
            axes[0].add_patch(rectangle1)
            axes[1].add_patch(rectangle2)

    save_figure(fig, dpi, dir_path_img, fname_img)

    save_figure(fig, dpi, "", "joint_plot.png")

    # close the plot handle
    plt.close()


def compute_and_save_jacobian_visualization(model, device, X_state_train, dpi, n_one_direction, dir_path_img, fname_img):

    X_state_train = X_state_train.detach().cpu()

    alpha = 0.5
    alpha_train_samples = 0.25

    dimX = np.linspace(LIMITS_HEATMAP[0][0], LIMITS_HEATMAP[0][1], n_one_direction)
    dimY = np.linspace(LIMITS_HEATMAP[1][0], LIMITS_HEATMAP[1][1], n_one_direction)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(np.stack((dimX.flatten(), dimY.flatten()), axis = -1), requires_grad = True).to(device)

    jac = torch.zeros(size = (n_one_direction*n_one_direction, N_TRAJOPT*N_DIM_THETA, N_DIM_X))

    for i in range(n_one_direction*n_one_direction) :
        jac[i] = torch.reshape(torch.autograd.functional.jacobian(model, x_state[i:i+1], create_graph=False, strict=False), shape = (N_TRAJOPT*N_DIM_THETA, N_DIM_X))

    jac_norm = torch.reshape(jac, shape = (n_one_direction, n_one_direction, N_TRAJOPT*N_DIM_THETA*N_DIM_X))
    jac_norm = torch.norm(jac_norm, p = "fro", dim = -1)
    jac_norm = jac_norm.detach().cpu()
    jac_norm_min = torch.min(jac_norm) 
    jac_norm_max = torch.max(jac_norm)

    # plot

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=1, hspace=1)

    ax.set_aspect(aspect = 'equal', adjustable = 'box')

    ax.set_title(
        '\nJacobian Frobenius Norm Landscape\n2D Two-Linkage Robot Inverse Kinematics\n',
        fontdict = {'fontsize': 15, 'fontweight': 'normal', 'horizontalalignment': 'center'},
        pad = 5
    )

    ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
    c = ax.pcolormesh(dimX, dimY, jac_norm, cmap = 'RdBu', shading = 'gouraud', norm = matplotlib.colors.LogNorm(vmin = jac_norm_min, vmax = jac_norm_max))

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], ms = 2.0, marker = 'o', color = 'k', ls = '', alpha = alpha_train_samples)

    cb = fig.colorbar(c, ax = ax, extend = 'max')

    if SAMPLE_CIRCLE :
        circleInner = plt.Circle((0.0, 0.0), radius = RADIUS_INNER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)
        circleOuter = plt.Circle((0.0, 0.0), radius = RADIUS_OUTER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)

    if LIMITS_HEATMAP != LIMITS :
        rectangle = plt.Rectangle(xy = (LIMITS[0][0], LIMITS[1][0]), width = LIMITS[0][1]-LIMITS[0][0], height = LIMITS[1][1]-LIMITS[1][0], color = 'orange', fill = False, lw = 4.0, alpha = alpha)

    save_figure(fig, dpi, dir_path_img, "no_train_region_" + fname_img)

    legend_entries = []

    if LIMITS_HEATMAP != LIMITS or SAMPLE_CIRCLE:

        legend_entries = legend_entries + [matplotlib.patches.Patch(color = 'orange', alpha = alpha, label = 'Sampling Area')]

        if SAMPLE_CIRCLE :
            ax.add_patch(circleInner)
            ax.add_patch(circleOuter)

        if LIMITS_HEATMAP != LIMITS:
            ax.add_patch(rectangle)

    plt.legend(loc = 'upper right', handles = legend_entries)

    save_figure(fig, dpi, dir_path_img, fname_img)

    save_figure(fig, dpi, "", "jacobian_visualization.png")
 
    # close the plot handle
    plt.close()


def compute_and_save_heatmap_plot(model, device, X_state_train, metrics_test, dpi, n_one_direction_heatmap, dir_path_img, fname_img):

    X_state_train = X_state_train.detach().cpu()

    test_terminal_energy_mean = metrics_test[0].detach().cpu()
    test_terminal_energy_std = metrics_test[1].detach().cpu()

    alpha = 0.5
    alpha_train_samples = 0.25

    dimX = np.linspace(LIMITS_HEATMAP[0][0], LIMITS_HEATMAP[0][1], n_one_direction_heatmap)
    dimY = np.linspace(LIMITS_HEATMAP[1][0], LIMITS_HEATMAP[1][1], n_one_direction_heatmap)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(np.stack((dimX.flatten(), dimY.flatten()), axis = -1)).to(device)

    terminal_energy = torch.zeros((x_state.shape[0])).to(device)

    with torch.no_grad() :

        if n_one_direction_heatmap > 100 :

            n_splits = 100

            delta = n_one_direction_heatmap*n_one_direction_heatmap // n_splits

            for split in range(n_splits):
                energy_tmp, constraint_tmp, terminal_position_distance_tmp, _ = compute_energy(model, x_state[split*delta:(split+1)*delta])
                terminal_energy[split*delta:(split+1)*delta] = terminal_position_distance_tmp #energy_tmp

        else :

            energy, constraint, terminal_position_distance, _ = compute_energy(model, x_state)
            terminal_energy = terminal_position_distance

    terminal_energy = np.array(terminal_energy.detach().cpu().reshape((n_one_direction_heatmap, n_one_direction_heatmap)).tolist())

    terminal_energy_min = terminal_energy.min()
    terminal_energy_max = terminal_energy.max()

    # plot

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=1, hspace=1)

    ax.set_aspect(aspect = 'equal', adjustable = 'box')

    ax.set_title(
        '\nTerminal Energy Landscape in Meters\n2D Two-Linkage Robot Inverse Kinematics\n',
        fontdict = {'fontsize': 15, 'fontweight': 'normal', 'horizontalalignment': 'center'},
        pad = 5
    )

    ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
    c = ax.pcolormesh(dimX, dimY, terminal_energy, cmap = 'RdBu', shading = 'gouraud', norm = matplotlib.colors.LogNorm(vmin = terminal_energy_min, vmax = terminal_energy_max))

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], ms = 2.0, marker = 'o', color = 'k', ls = '', alpha = alpha_train_samples)

    cb = fig.colorbar(c, ax = ax, extend = 'max')
    cb.ax.plot([0, 1], [test_terminal_energy_mean]*2, 'k', alpha = alpha, lw = 8.0)
    cb.ax.plot([0, 1], [test_terminal_energy_mean + test_terminal_energy_std]*2, 'k', alpha = alpha, lw = 3.0)
    cb.ax.plot([0, 1], [test_terminal_energy_mean - test_terminal_energy_std]*2, 'k', alpha = alpha, lw = 3.0)

    if SAMPLE_CIRCLE :
        circleInner = plt.Circle((0.0, 0.0), radius = RADIUS_INNER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)
        circleOuter = plt.Circle((0.0, 0.0), radius = RADIUS_OUTER, color = 'orange', fill = False, lw = 4.0, alpha = alpha)

    if LIMITS_HEATMAP != LIMITS :
        rectangle = plt.Rectangle(xy = (LIMITS[0][0], LIMITS[1][0]), width = LIMITS[0][1]-LIMITS[0][0], height = LIMITS[1][1]-LIMITS[1][0], color = 'orange', fill = False, lw = 4.0, alpha = alpha)

    save_figure(fig, dpi, dir_path_img, "no_train_region_" + fname_img)

    legend_entries = [
            matplotlib.lines.Line2D([0], [0], lw = 0.0, marker = 'o', color = 'k', alpha = alpha_train_samples, markersize = 10.0, label = 'Train Samples'),
            matplotlib.patches.Patch(color = 'k', alpha = alpha, label = 'Test Mean ± Std')
        ]

    if LIMITS_HEATMAP != LIMITS or SAMPLE_CIRCLE:

        legend_entries = legend_entries + [matplotlib.patches.Patch(color = 'orange', alpha = alpha, label = 'Sampling Area')]

        if SAMPLE_CIRCLE :
            ax.add_patch(circleInner)
            ax.add_patch(circleOuter)

        if LIMITS_HEATMAP != LIMITS:
            ax.add_patch(rectangle)

    plt.legend(loc = 'upper right', handles = legend_entries)

    save_figure(fig, dpi, dir_path_img, fname_img)

    save_figure(fig, dpi, "", "heatmap.png")

    # transform the final plot into an array to save on tensorboard

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches = "tight", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    # close the plot handle
    plt.close()

    return img_arr


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

compute_and_save_samples_plot(X_state_train_all.detach().cpu(), X_state_val.detach().cpu(), X_state_test.detach().cpu(), dir_path_id_img_samples, "samples_plot.jpg")

print("\nTraining Starts!\n")

time_measure = 0
nb_actual_iterations = 0
diffs = []

X_state_train = 0
distances = 0
distances_indices_sorted = 0
distance_index = 0

for j in range(N_ITERATIONS) :

    tic_loop = time.perf_counter()

    nb_actual_iterations += 1
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


    is_visualize = True if nb_actual_iterations % FK_VISUALIZATION_UPDATE == 0 else False

    [loss_train, metrics_train] = compute_loss(model, X_state_train, nb_actual_iterations, is_visualize, "train", dir_path_id_img_train)

    optimizer.zero_grad()
    loss_train.backward()
    # prevent potential exploding gradients
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()
    scheduler.step()

    if nb_actual_iterations % TENSORBOARD_UPDATE == 0 or j == 0 or j == N_ITERATIONS - 1 :

        loss_val = 0
        loss_test = 0
        metrics_val = []
        metrics_test = []
        dloss_train_dW = 0

        with torch.no_grad() :

            dloss_train_dW = compute_dloss_dW(model)

            [loss_val, metrics_val] = compute_loss(model, X_state_val, nb_actual_iterations, is_visualize, "val", dir_path_id_img_val)

            tb_writer.add_scalar('Learning Rate', current_lr, nb_actual_iterations)
            tb_writer.add_scalar('Train Loss', loss_train.detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Mean Train Terminal Position Distance [m]', metrics_train[0].detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Stddev Train Terminal Position Distance [m]', metrics_train[1].detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Max Train Terminal Position Distance [m]', metrics_train[2].detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Val Loss', loss_val.detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Mean Val Terminal Position Distance [m]', metrics_val[0].detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Stddev Val Terminal Position Distance [m]', metrics_val[1].detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Max Val Terminal Position Distance [m]', metrics_val[2].detach().cpu(), nb_actual_iterations)
            tb_writer.add_scalar('Loss Gradient Norm', dloss_train_dW, nb_actual_iterations)

            if j == N_ITERATIONS - 1 :

                [loss_test, metrics_test] = compute_loss(model, X_state_test, nb_actual_iterations, is_visualize, "test", dir_path_id_img_test)

                tb_writer.add_scalar('Test Loss', loss_test.detach().cpu(), nb_actual_iterations)
                tb_writer.add_scalar('Mean Test Terminal Position Distance [m]', metrics_test[0].detach().cpu(), nb_actual_iterations)
                tb_writer.add_scalar('Stddev Test Terminal Position Distance [m]', metrics_test[1].detach().cpu(), nb_actual_iterations)
                tb_writer.add_scalar('Max Test Terminal Position Distance [m]', metrics_test[2].detach().cpu(), nb_actual_iterations)

        heatmap_dpi = SAVEFIG_DPI_FINAL if j == N_ITERATIONS - 1 else SAVEFIG_DPI
        n_one_direction_heatmap = 1000 if j == N_ITERATIONS - 1 else 50
        jacobian_visualization_name = "jacobian_visualization_final_{}.png".format(nb_actual_iterations) if j == N_ITERATIONS - 1 else "jacobian_visualization_{}.png".format(nb_actual_iterations)
        joint_plot_name = "joint_plot_final_{}.png".format(nb_actual_iterations) if j == N_ITERATIONS - 1 else "joint_plot_{}.png".format(nb_actual_iterations)
        heatmap_name = "heatmap_final_{}.png".format(nb_actual_iterations) if j == N_ITERATIONS - 1 else "heatmap_{}.png".format(nb_actual_iterations)

        tic = time.perf_counter()

        compute_and_save_jacobian_visualization(model, device, X_state_train, heatmap_dpi, min(200, n_one_direction_heatmap), dir_path_id_jacobian_visualization, jacobian_visualization_name)
        compute_and_save_joint_plot(model, device, heatmap_dpi, n_one_direction_heatmap, dir_path_id_joint_plot, joint_plot_name)

        heatmap_metric = metrics_val
        if j == N_ITERATIONS - 1 :
            heatmap_metric = metrics_test
            
        image_tensor = compute_and_save_heatmap_plot(model, device, X_state_train, heatmap_metric, heatmap_dpi, n_one_direction_heatmap, dir_path_id_heatmap, heatmap_name)
        
        toc = time.perf_counter()

        print(f"{toc - tic:0.2f} [s] for image_tensor = compute_and_save_heatmap_plot(...)")

        tb_writer.add_image("heatmap", img_tensor = image_tensor, global_step = nb_actual_iterations, dataformats = 'HWC')

    toc_loop = time.perf_counter()
    time_measure_tmp = (toc_loop - tic_loop)
    time_measure += time_measure_tmp

    if nb_actual_iterations % TIME_MEASURE_UPDATE == 0 :
        print(f"{nb_actual_iterations} iterations {time_measure_tmp:0.2f} [s] (total {time_measure:0.2f} [s])")

print("\nTraining Process Completed.\n")

save_model(model, nb_actual_iterations, dir_path_id_model, nn_model_state_dict_only_str, nn_model_full_str)

print("\nAll Done!\n")

sys.stdout = sys_stdout_original
file_handle_logger.close()