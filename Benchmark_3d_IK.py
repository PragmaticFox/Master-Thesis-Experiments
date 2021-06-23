# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#!/bin/python3

import os
import sys
import math
import time
import shutil
import random
import pathlib

import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

DTYPE_NUMPY = np.float32
DTYPE_TORCH = torch.float32

torch.set_default_dtype(DTYPE_TORCH)

random.seed(42)
np.random.seed(84)
torch.manual_seed(168)
#torch.set_deterministic(True)
#torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

if True and torch.cuda.is_available() :

    device = "cuda:0" 
    print("CUDA is available! Computing on GPU.")

else :

    device = "cpu"  
    print("CUDA is unavailable! Computing on CPU.")

device = torch.device(device)

N_TRAJOPT = 1
N_DIM_THETA = 12
N_DIM_X_STATE = 6

ORIGIN = [0.0, 0.0, 0.0]

PX = torch.tensor([1.0, 0.0, 0.0]).to(device)
PY = torch.tensor([0.0, 1.0, 0.0]).to(device)
PZ = torch.tensor([0.0, 0.0, 1.0]).to(device)


# %%
class Model(torch.nn.Module):

    # U-Net as described in
    # https://arxiv.org/pdf/1808.03856v5.pdf
    
    ''' DO NOT CHANGE '''

    def __init__(self):

        max_neurons = 256 

        super(Model, self).__init__()

        self.fc_in = torch.nn.Linear(6, max_neurons)

        self.fc_down_1 = torch.nn.Linear(max_neurons, max_neurons // 2)
        self.fc_down_2 = torch.nn.Linear(max_neurons // 2, max_neurons // 4)
        self.fc_down_3 = torch.nn.Linear(max_neurons // 4, max_neurons // 8)
        self.fc_down_4 = torch.nn.Linear(max_neurons // 8, max_neurons // 8)
        self.fc_up_4 = torch.nn.Linear(max_neurons // 8, max_neurons // 8)
        self.fc_up_3 = torch.nn.Linear(max_neurons // 8, max_neurons // 4)
        self.fc_up_2 = torch.nn.Linear(max_neurons // 4, max_neurons // 2)
        self.fc_up_1 = torch.nn.Linear(max_neurons // 2, max_neurons)

        self.fc_out = torch.nn.Linear(max_neurons, 2*N_DIM_THETA)

        self.BIAS = torch.tensor([4.929186, -15.617743, -94.050809, 17.956906, 87.800833, -2.386771]).to(device) * math.pi/180.0

    def forward(self, x):

        x = self.fc_in(x)
        x = torch.relu_(x)

        down_1 = self.fc_down_1(x)
        down_1 = torch.relu_(down_1)

        down_2 = self.fc_down_2(down_1)
        down_2 = torch.relu_(down_2)

        down_3 = self.fc_down_3(down_2)
        down_3 = torch.relu_(down_3)

        down_4 = self.fc_down_4(down_3)
        down_4 = torch.relu_(down_4)

        up_3 = self.fc_up_4(down_4)
        up_3 = torch.relu_(up_3 + down_3)

        up_2 = self.fc_up_3(up_3)
        up_2 = torch.relu_(up_2 + down_2)

        up_1 = self.fc_up_2(up_2)
        up_1 = torch.relu_(up_1 + down_1)

        x = self.fc_up_1(up_1)
        x = torch.relu_(x)

        x = self.fc_out(x)

        x = torch.reshape(x, shape = (x.shape[0], N_DIM_THETA, 2))

        # theta = arctan(y/x) = arctan(1.0*sin(theta), 1.0*cos(theta))
        theta = torch.atan2(x[:, :, 0], x[:,:, 1]) 
        theta = theta + self.BIAS

        return theta


# %%
def dh_matrix(n, theta, alpha, d, r) :

    device = alpha.device

    transform = torch.reshape(torch.eye(4,4), shape = (1, 4, 4)).repeat(n, 1, 1).to(device)

    transform[:, 0, 0] = torch.cos(theta)
    transform[:, 0, 1] = - torch.sin(theta) * torch.cos(alpha)
    transform[:, 0, 2] = torch.sin(theta) * torch.sin(alpha)
    transform[:, 0, 3] = r * torch.cos(theta)

    transform[:, 1, 0] = torch.sin(theta)
    transform[:, 1, 1] = torch.cos(theta) * torch.cos(alpha)
    transform[:, 1, 2] = - torch.cos(theta) * torch.sin(alpha)
    transform[:, 1, 3] = r * torch.sin(theta)

    transform[:, 2, 1] = torch.sin(alpha)
    transform[:, 2, 2] = torch.cos(alpha)
    transform[:, 2, 3] = d

    return transform


def fk(theta):

    device = theta.device
    nn = theta.shape[0] # n_batch_times_n_trajOpt
    n_dim_theta = theta.shape[1]

    transform0 = torch.reshape(torch.eye(4,4), shape = (1, 4, 4)).repeat(nn, 1, 1).to(device)

    transform1 = torch.matmul(transform0, dh_matrix(nn, theta[:, 0], theta[:, 1], 0.089159, 0.0))
    transform2 = torch.matmul(transform1, dh_matrix(nn, theta[:, 2], theta[:, 3], 0.0, -0.425))
    transform3 = torch.matmul(transform2, dh_matrix(nn, theta[:, 4], theta[:, 5], 0.0, -0.39225))
    transform4 = torch.matmul(transform3, dh_matrix(nn, theta[:, 6], theta[:, 7], 0.10915, 0.0))
    transform5 = torch.matmul(transform4, dh_matrix(nn, theta[:, 8], theta[:, 9], 0.09465, 0.0))
    transform6 = torch.matmul(transform5, dh_matrix(nn, theta[:, 10], theta[:, 11], 0.0823, 0.0))

    p = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    p_final = torch.reshape(torch.tensor([0.0, 0.0, 0.0, 1.0]), shape = (1, 1, 4)).repeat(nn, n_dim_theta//2, 1).to(device)

    transform = [transform1, transform2, transform3, transform4, transform5, transform6]

    for i in range(n_dim_theta//2) :
            
        p_final[:, i] = torch.matmul(transform[i], p)

    return p_final[:, :, :-1], transform6[:, :3, :3]


# %%
def compute_orientation_matrix(orientation_vector) :

    n_batch = orientation_vector.shape[0]
    device = orientation_vector.device

    # orientation vector is in euler angles
    # phi = 0th entry 
    # theta = 1st entry
    # psi = 2nd entry 

    c_phi = torch.cos(orientation_vector[:, 0])
    c_theta = torch.cos(orientation_vector[:, 1])
    c_psi = torch.cos(orientation_vector[:, 2])

    s_phi = torch.sin(orientation_vector[:, 0])
    s_theta = torch.sin(orientation_vector[:, 1])
    s_psi = torch.sin(orientation_vector[:, 2])

    orientation_matrix = torch.reshape(torch.eye(3, 3), shape = (1, 3, 3)).repeat(n_batch, 1, 1).to(device)

    orientation_matrix[:, 0, 0] = c_theta * c_psi
    orientation_matrix[:, 0, 1] = - c_phi * s_psi + s_phi * s_theta * c_psi
    orientation_matrix[:, 0, 2] = s_phi * s_psi + c_phi * s_theta * c_psi

    orientation_matrix[:, 1, 0] = c_theta * s_psi
    orientation_matrix[:, 1, 1] = c_phi * c_psi + s_phi * s_theta * s_psi
    orientation_matrix[:, 1, 2] = - s_phi * c_psi + c_phi * s_theta * s_psi

    orientation_matrix[:, 2, 0] = - s_theta
    orientation_matrix[:, 2, 1] = s_phi * c_theta
    orientation_matrix[:, 2, 2] = c_phi * c_theta

    return orientation_matrix


def save_figure(figure, dir_path_img, fname_img):

    figure.savefig(
        fname = pathlib.Path(dir_path_img, fname_img),
        bbox_inches = "tight",
        dpi = 250,
        #pil_kwargs = {'optimize': True, 'quality': 75}
    )


def visualize_end_effector_and_save(
    x_end_effector_position,
    x_end_effector_orientation_positions,
    x_hat_end_effector_position,
    x_hat_end_effector_orientation_positions,
    x_hat_positions,
    dir_path_img,
    fname_img
):

    quiver_length = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ORIGIN[0], ORIGIN[1], ORIGIN[2], c = 'k', s = 100, zorder = 1)

    ax.scatter(x_end_effector_position[0], x_end_effector_position[1], x_end_effector_position[2], c = 'k', s = 75, zorder = -10, alpha = 0.5)

    ax.scatter(x_end_effector_orientation_positions[0][0], x_end_effector_orientation_positions[0][1], x_end_effector_orientation_positions[0][2], c = 'g', s = 75, zorder = -10, alpha = 0.5)
    ax.scatter(x_end_effector_orientation_positions[1][0], x_end_effector_orientation_positions[1][1], x_end_effector_orientation_positions[1][2], c = 'b', s = 75, zorder = -10, alpha = 0.5)
    ax.scatter(x_end_effector_orientation_positions[2][0], x_end_effector_orientation_positions[2][1], x_end_effector_orientation_positions[2][2], c = 'r', s = 75, zorder = -10, alpha = 0.5)

    ax.quiver(
        x_end_effector_position[0],
        x_end_effector_position[1],
        x_end_effector_position[2],
        x_end_effector_orientation_positions[0][0] - x_end_effector_position[0],
        x_end_effector_orientation_positions[0][1] - x_end_effector_position[1],
        x_end_effector_orientation_positions[0][2] - x_end_effector_position[2],
        color = 'g',
        length = quiver_length,
        alpha = 0.5,
        lw = 1.5
    )

    ax.quiver(
        x_end_effector_position[0],
        x_end_effector_position[1],
        x_end_effector_position[2],
        x_end_effector_orientation_positions[1][0] - x_end_effector_position[0],
        x_end_effector_orientation_positions[1][1] - x_end_effector_position[1],
        x_end_effector_orientation_positions[1][2] - x_end_effector_position[2],
        color = 'b',
        length = quiver_length,
        alpha = 0.5,
        lw = 1.5
    )

    ax.quiver(
        x_end_effector_position[0],
        x_end_effector_position[1],
        x_end_effector_position[2],
        x_end_effector_orientation_positions[2][0] - x_end_effector_position[0],
        x_end_effector_orientation_positions[2][1] - x_end_effector_position[1],
        x_end_effector_orientation_positions[2][2] - x_end_effector_position[2],
        color = 'r',
        length = quiver_length,
        alpha = 0.5,
        lw = 1.5
    )

    ax.quiver(
        ORIGIN[0],
        ORIGIN[1],
        ORIGIN[2],
        x_end_effector_position[0] - ORIGIN[0],
        x_end_effector_position[1] - ORIGIN[1],
        x_end_effector_position[2] - ORIGIN[2],
        color = '0.5',
        length = quiver_length,
        alpha = 0.5,
        lw = 1.5
    )

    ax.scatter(x_hat_end_effector_position[-1][0], x_hat_end_effector_position[-1][1], x_hat_end_effector_position[-1][2], c = 'k', s = 50, zorder = 100)

    ax.scatter(x_hat_end_effector_orientation_positions[0][0], x_hat_end_effector_orientation_positions[0][1], x_hat_end_effector_orientation_positions[0][2], c = 'g', s = 50, zorder = 100, ec = 'k')
    ax.scatter(x_hat_end_effector_orientation_positions[1][0], x_hat_end_effector_orientation_positions[1][1], x_hat_end_effector_orientation_positions[1][2], c = 'b', s = 50, zorder = 100, ec = 'k')
    ax.scatter(x_hat_end_effector_orientation_positions[2][0], x_hat_end_effector_orientation_positions[2][1], x_hat_end_effector_orientation_positions[2][2], c = 'r', s = 50, zorder = 100, ec = 'k')

    ax.quiver(
        x_hat_end_effector_position[-1][0],
        x_hat_end_effector_position[-1][1],
        x_hat_end_effector_position[-1][2],
        x_hat_end_effector_orientation_positions[0][0] - x_hat_end_effector_position[-1][0],
        x_hat_end_effector_orientation_positions[0][1] - x_hat_end_effector_position[-1][1],
        x_hat_end_effector_orientation_positions[0][2] - x_hat_end_effector_position[-1][2],
        color = 'g',
        length = quiver_length
    )

    ax.quiver(
        x_hat_end_effector_position[-1][0],
        x_hat_end_effector_position[-1][1],
        x_hat_end_effector_position[-1][2],
        x_hat_end_effector_orientation_positions[1][0] - x_hat_end_effector_position[-1][0],
        x_hat_end_effector_orientation_positions[1][1] - x_hat_end_effector_position[-1][1],
        x_hat_end_effector_orientation_positions[1][2] - x_hat_end_effector_position[-1][2],
        color = 'b',
        length = quiver_length
    )

    ax.quiver(
        x_hat_end_effector_position[-1][0],
        x_hat_end_effector_position[-1][1],
        x_hat_end_effector_position[-1][2],
        x_hat_end_effector_orientation_positions[2][0] - x_hat_end_effector_position[-1][0],
        x_hat_end_effector_orientation_positions[2][1] - x_hat_end_effector_position[-1][1],
        x_hat_end_effector_orientation_positions[2][2] - x_hat_end_effector_position[-1][2],
        color = 'r',
        length = quiver_length
    )

    ax.quiver(
        ORIGIN[0],
        ORIGIN[1],
        ORIGIN[2],
        x_hat_end_effector_position[-1][0] - ORIGIN[0],
        x_hat_end_effector_position[-1][1] - ORIGIN[1],
        x_hat_end_effector_position[-1][2] - ORIGIN[2],
        color = '0.5',
        length = quiver_length
    )

    for t in range(N_TRAJOPT-1) :

        ax.scatter(x_hat_end_effector_position[t][0], x_hat_end_effector_position[t][1], x_hat_end_effector_position[t][2], c = 'k', s = 25, zorder = 100)

    for t in range(N_TRAJOPT) :

        ax.quiver(
            ORIGIN[0],
            ORIGIN[1],
            ORIGIN[2],
            x_hat_positions[0][t][0] - ORIGIN[0],
            x_hat_positions[0][t][1] - ORIGIN[1],
            x_hat_positions[0][t][2] - ORIGIN[2],
            color = 'b',
            alpha = 0.8,
            lw = 2,
            arrow_length_ratio = 0.4
        )

        for i in range(1, len(x_hat_positions), 1) :

            ax.scatter(x_hat_positions[i-1][t][0], x_hat_positions[i-1][t][1], x_hat_positions[i-1][t][2], c = '0.5', s = 20)

            ax.quiver(
                x_hat_positions[i-1][t][0],
                x_hat_positions[i-1][t][1],
                x_hat_positions[i-1][t][2],
                x_hat_positions[i][t][0] - x_hat_positions[i-1][t][0],
                x_hat_positions[i][t][1] - x_hat_positions[i-1][t][1],
                x_hat_positions[i][t][2] - x_hat_positions[i-1][t][2],
                color = 'b',
                alpha = 0.8,
                lw = 2,
                arrow_length_ratio = 0.4
            )

        ax.scatter(x_hat_positions[-1][t][0], x_hat_positions[-1][t][1], x_hat_positions[-1][t][2], c = 'k', s = 25, zorder = 100)


    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    #ax.set_zlim(-1, 1)

    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)

    plt.gca().set_aspect('auto', adjustable='box')

    save_figure(plt.gcf(), dir_path_img, fname_img)

    plt.close()



def compute_loss(model, x_state, iteration_index, fname):

    n_batch = x_state.shape[0]
    device = x_state.device

    theta_hat = model(x_state)

    theta_hat_reshaped = torch.reshape(input = theta_hat, shape = (n_batch*N_TRAJOPT, N_DIM_THETA))

    x_hat_fk_chain, orientation_matrix_hat = fk(theta_hat_reshaped)

    x_hat_fk_chain = torch.reshape(input = x_hat_fk_chain, shape = (n_batch, N_TRAJOPT, N_DIM_THETA//2, 3))
    x_hat_fk_chain = torch.transpose(input = x_hat_fk_chain, dim0 = 1, dim1 = 2)

    orient_hat_x = torch.matmul(orientation_matrix_hat, PX) + x_hat_fk_chain[:, -1, -1, :]
    orient_hat_y = torch.matmul(orientation_matrix_hat, PY) + x_hat_fk_chain[:, -1, -1, :]
    orient_hat_z = torch.matmul(orientation_matrix_hat, PZ) + x_hat_fk_chain[:, -1, -1, :]

    orientation_matrix = compute_orientation_matrix(x_state[:, 3:])
    orient_x = torch.matmul(orientation_matrix, PX) + x_state[:, :3]
    orient_y = torch.matmul(orientation_matrix, PY) + x_state[:, :3]
    orient_z = torch.matmul(orientation_matrix, PZ) + x_state[:, :3]

    terminal_orientation_distance_px = torch.norm((orient_hat_x - orient_x), p = 2.0, dim = -1)
    terminal_orientation_distance_py = torch.norm((orient_hat_y - orient_y), p = 2.0, dim = -1)
    terminal_orientation_distance_pz = torch.norm((orient_hat_z - orient_z), p = 2.0, dim = -1)
    terminal_position_distance = torch.norm((x_state[:, :3] - x_hat_fk_chain[:, -1, -1, :]), p = 2.0, dim = -1)

    energy_batch = torch.pow(terminal_position_distance, exponent = 2.0)
    energy_batch += torch.pow(terminal_orientation_distance_px, exponent = 2.0)
    energy_batch += torch.pow(terminal_orientation_distance_py, exponent = 2.0)
    energy_batch += torch.pow(terminal_orientation_distance_pz, exponent = 2.0)

    loss = torch.mean(energy_batch)

    metric0 = torch.mean(terminal_position_distance)
    metric1 = (torch.mean(terminal_orientation_distance_px) + torch.mean(terminal_orientation_distance_py) + torch.mean(terminal_orientation_distance_pz)) / 3.0

    if iteration_index % 100 == 0 :

        index_train_batch_worst = np.argmax(energy_batch.detach().tolist())
        index_train_batch_random = random.randint(0, n_batch-1)

        visualize_end_effector_and_save(
            x_state[index_train_batch_worst, :3].tolist(),
            [orient_x[index_train_batch_worst].tolist(), orient_y[index_train_batch_worst].tolist(), orient_z[index_train_batch_worst].tolist()],
            x_hat_fk_chain[index_train_batch_worst, -1].tolist(),
            [orient_hat_x[index_train_batch_worst].tolist(), orient_hat_y[index_train_batch_worst].tolist(), orient_hat_z[index_train_batch_worst].tolist()],
            x_hat_fk_chain[index_train_batch_worst].tolist(),
            "trajectory_optimization_files",
            fname + "_{}.jpg".format(0)
        )

    return loss, [metric0, metric1]


# %%
LIMITS = [
    [0.2, 0.8],
    [-0.3, 0.3],
    [-0.3, 0.3]
]

FIXED_ORIENTATION = [-1.529975, 2.644308e-2, 1.699597]

change = 0.0#math.pi/10
FIXED_ORIENTATION_LIMITS = [
    [ FIXED_ORIENTATION[0] - change, FIXED_ORIENTATION[0] + change],
    [ FIXED_ORIENTATION[1] - change, FIXED_ORIENTATION[1] + change],
    [ FIXED_ORIENTATION[2] - change, FIXED_ORIENTATION[2] + change],
]

num_samples_train = 10000

X_train = torch.rand(num_samples_train, 6).to(device)

X_train[:, 0] = LIMITS[0][0] + X_train[:, 0] * ( LIMITS[0][1] - LIMITS[0][0] )
X_train[:, 1] = LIMITS[1][0] + X_train[:, 1] * ( LIMITS[1][1] - LIMITS[1][0] )
X_train[:, 2] = LIMITS[2][0] + X_train[:, 2] * ( LIMITS[2][1] - LIMITS[2][0] )

X_train[:, 3] = FIXED_ORIENTATION_LIMITS[0][0] + X_train[:, 3] * ( FIXED_ORIENTATION_LIMITS[0][1] - FIXED_ORIENTATION_LIMITS[0][0] )
X_train[:, 4] = FIXED_ORIENTATION_LIMITS[1][0] + X_train[:, 4] * ( FIXED_ORIENTATION_LIMITS[1][1] - FIXED_ORIENTATION_LIMITS[1][0] )
X_train[:, 5] = FIXED_ORIENTATION_LIMITS[2][0] + X_train[:, 5] * ( FIXED_ORIENTATION_LIMITS[2][1] - FIXED_ORIENTATION_LIMITS[2][0] )

num_samples_val = 1000

X_val = torch.rand(num_samples_val, 6).to(device)

X_val[:, 0] = LIMITS[0][0] + X_val[:, 0] * ( LIMITS[0][1] - LIMITS[0][0] )
X_val[:, 1] = LIMITS[1][0] + X_val[:, 1] * ( LIMITS[1][1] - LIMITS[1][0] )
X_val[:, 2] = LIMITS[2][0] + X_val[:, 2] * ( LIMITS[2][1] - LIMITS[2][0] )

X_val[:, 3] = FIXED_ORIENTATION_LIMITS[0][0] + X_val[:, 3] * ( FIXED_ORIENTATION_LIMITS[0][1] - FIXED_ORIENTATION_LIMITS[0][0] )
X_val[:, 4] = FIXED_ORIENTATION_LIMITS[1][0] + X_val[:, 4] * ( FIXED_ORIENTATION_LIMITS[1][1] - FIXED_ORIENTATION_LIMITS[1][0] )
X_val[:, 5] = FIXED_ORIENTATION_LIMITS[2][0] + X_val[:, 5] * ( FIXED_ORIENTATION_LIMITS[2][1] - FIXED_ORIENTATION_LIMITS[2][0] )

model = Model().to(device)
tb_writer = SummaryWriter()

N_BATCH = 100

for i in range(1) :

    indices = random.sample(range(0, num_samples_train), N_BATCH)
    x_state_batch = X_train[indices]

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda epoch: 0.9999)

    for j in range(100000) :
        
        tb_writer.add_scalar('LR', optimizer.param_groups[0]['lr'], j)

        loss_train, metrics_train = compute_loss(model, x_state_batch, j, "train")

        tb_writer.add_scalar('Train Loss', loss_train.detach().cpu(), j)
        tb_writer.add_scalar('Train Terminal Position Distance [m]', metrics_train[0].detach().cpu(), j)
        tb_writer.add_scalar('Train Terminal Orientation Distance [m]', metrics_train[1].detach().cpu(), j)

        optimizer.zero_grad()
        loss_train.backward()
        # exploding gradients are never useful
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad() :

            loss_val, metrics_val = compute_loss(model, X_val, j, "val")

            tb_writer.add_scalar('Val Loss', loss_val.detach().cpu(), j)
            tb_writer.add_scalar('Val Terminal Position Distance [m]', metrics_val[0].detach().cpu(), j)
            tb_writer.add_scalar('Val Terminal Orientation Distance [m]', metrics_val[1].detach().cpu(), j)

