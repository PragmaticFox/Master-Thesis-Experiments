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
import scipy
import scipy.stats


from torch.utils.tensorboard import SummaryWriter

DTYPE_NUMPY = np.float64
DTYPE_TORCH = torch.float64

torch.set_default_dtype(DTYPE_TORCH)

#SAVEFIG_DPI = 100
SAVEFIG_DPI = 500

random.seed(412)
np.random.seed(284)
torch.manual_seed(1468)
#torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False
#torch.autograd.set_detect_anomaly(True)

identifier_string = "Benchmark_2d_IK_joint_limits"

log_file_str = "train_eval_log_file.txt"

nn_model_full_str = "nn_model_full"
nn_model_state_dict_only_str = "nn_model_state_dict_only"

dir_path_id = pathlib.Path("D:/trajectory_optimization/master_thesis_experiments/", identifier_string)

dir_path_id_data = pathlib.Path(dir_path_id, "data")
dir_path_id_plot = pathlib.Path(dir_path_id, "plot")
dir_path_id_model = pathlib.Path(dir_path_id, "model")

dir_path_id_img_val = pathlib.Path(dir_path_id, "img_val")
dir_path_id_img_train = pathlib.Path(dir_path_id, "img_train")
dir_path_id_img_test = pathlib.Path(dir_path_id, "img_test")
dir_path_id_img_samples = pathlib.Path(dir_path_id, "img_samples")

N_SAMPLES_TRAIN = 1
N_SAMPLES_VAL = 1000
N_SAMPLES_TEST = 10000

N_DIM_THETA = 2
N_DIM_X = 2
N_DIM_X_STATE = 1*N_DIM_X
N_TRAJOPT = 1
N_ITERATIONS = 50000

NN_DIM_IN = 1*N_DIM_X_STATE
NN_DIM_OUT = 2*N_DIM_THETA*N_TRAJOPT
NN_DIM_IN_TO_OUT = 256

FK_ORIGIN = [0.0, 0.0]

LIMITS = [
    [0.3999, 0.4],
    [0.3999, 0.4]
]

LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]

LR_INITIAL = 1e-3

LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.9999

TENSORBOARD_UPDATE = 250


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

    def __init__(self):

        super(Model, self).__init__()

        self.fc_start_1 = torch.nn.Linear(NN_DIM_IN, 1*NN_DIM_IN_TO_OUT)

        self.fc_middle = torch.nn.Linear(1*NN_DIM_IN_TO_OUT, 1*NN_DIM_IN_TO_OUT)

        self.fc_end = torch.nn.Linear(NN_DIM_IN_TO_OUT, NN_DIM_OUT)
        self.fc_end_alt = torch.nn.Linear(NN_DIM_IN_TO_OUT, NN_DIM_OUT // 2)

        #self.act = torch.nn.Softplus()
        self.act = torch.nn.ReLU()

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

    if not dir_path_id.exists() :

        dir_path_id.mkdir()

    if not dir_path_id_data.exists() :

        dir_path_id_data.mkdir()

    if not dir_path_id_plot.exists() :

        dir_path_id_plot.mkdir()

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


def save_figure(figure, dir_path_img, fname_img):

    figure.savefig(
        fname = pathlib.Path(dir_path_img, fname_img),
        bbox_inches = "tight",
        dpi = SAVEFIG_DPI,
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

    save_figure(plt.gcf(), dir_path_img, fname_img)

    plt.close()


def compute_and_save_samples_plot(X_state_train, X_state_val, X_state_test, dir_path_img, fname_img):

    plt.plot(X_state_train[:, 0], X_state_train[:, 1], ms = 1.0, marker = 'o', color = 'b', ls = '')
    plt.plot(X_state_val[:, 0], X_state_val[:, 1], ms = 1.0, marker = 'o', color = 'g', ls = '')
    plt.plot(X_state_test[:, 0], X_state_test[:, 1], ms = 1.0, marker = 'o', color = 'r', ls = '')

    plt.gca().set_aspect('equal', adjustable='box')

    save_figure(plt.gcf(), dir_path_img, fname_img)

    plt.close()


def compute_loss(model, x_state, iteration_index, is_visualize, fname, dir_path):

    n_batch = x_state.shape[0]

    theta_hat = model(x_state)

    theta_hat_reshaped = torch.reshape(input = theta_hat, shape = (n_batch*N_TRAJOPT, N_DIM_THETA))

    x_hat_fk_chain = fk(theta_hat_reshaped)

    x_hat_fk_chain = torch.reshape(input = x_hat_fk_chain, shape = (n_batch, N_TRAJOPT, N_DIM_THETA, N_DIM_X_STATE))
    x_hat_fk_chain = torch.transpose(input = x_hat_fk_chain, dim0 = 1, dim1 = 2)

    terminal_position_distance = torch.norm((x_state - x_hat_fk_chain[:, -1, -1, :]), p = 2, dim = -1)

    energy_batch = torch.pow(terminal_position_distance, exponent = 2)

    # is 0 if angle is between -pi and 0 (180° and 360°), otherwise between 0 and pi (penalty)
    #energy_batch += torch.threshold(theta_hat[:, -1], threshold = 0.0, value = 0.0)
    
    # Soft Upper Limit Constraint
    '''
    limit = math.pi
    epsilon = 1e-6 * math.pi # fabs(upper_limit - lower_limit) * relEps (upper_limit is either 0 or pi, lower_limit is either -pi or 0)
    stiffness = 1e-3

    x = math.pi + theta_hat[:, -1] # x is in [0, 2pi], represents the most outer joint angle of theta_hat in radians
    x = x - limit # x is in [-pi, pi] (just roll with this redundancy, lol)

    #print(x.shape)

    a1 = stiffness
    b1 = 0.5 * a1 * epsilon
    c1 = 1.0 / 6.0 * a1 * epsilon * epsilon

    a2 = 1.0 / (2.0 * epsilon) * a1
    b2 = a1
    c2 = 0.5 * a1 * epsilon
    d2 = 1.0 / 6.0 * a1 * epsilon * epsilon

    y = x[x > 0.0]

    y = 0.5 * a1 * y * y + b1 * y + c1

    z = x[x > -epsilon]
    z = z[z <= 0.0]

    z = 1.0 / 3.0 * a2 * z * z * z + 0.5 * b2 * z * z + c2 * z + d2

    x[x > -epsilon][x[x > -epsilon] <= 0.0] = z
    x[x > 0.0] = y
    x[x <= 0.0] = 0.0
    '''
    #print(x.shape)

    #energy_batch += x

    loss = torch.mean(energy_batch)

    metric0 = torch.mean(terminal_position_distance)
    metric1 = torch.std(terminal_position_distance)
    metric2 = torch.max(terminal_position_distance)

    if is_visualize :

        index_batch_worst = np.argmax(energy_batch.detach().tolist())

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
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = LR_INITIAL, total_steps=N_ITERATIONS, epochs=None, steps_per_epoch=1, pct_start=0.2, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1, verbose=False)

tb_writer = SummaryWriter()

X_state_train = torch.tensor([compute_sample() for _ in range(N_SAMPLES_TRAIN)], dtype = DTYPE_TORCH).to(device)
X_state_val = torch.tensor([compute_sample() for _ in range(N_SAMPLES_VAL)], dtype = DTYPE_TORCH).to(device)
X_state_test = torch.tensor([compute_sample() for _ in range(N_SAMPLES_TEST)], dtype = DTYPE_TORCH).to(device)

compute_and_save_samples_plot(X_state_train.detach().cpu(), X_state_val.detach().cpu(), X_state_test.detach().cpu(), dir_path_id_img_samples, "samples_plot.jpg")

print("\nTraining Starts!\n")

nb_actual_iterations = 0

for j in range(N_ITERATIONS) :
    
    nb_actual_iterations += 1
    current_lr = optimizer.param_groups[0]['lr']

    is_visualize = True if nb_actual_iterations % 10000 == 0 else False

    [loss_train, metrics_train] = compute_loss(model, X_state_train, nb_actual_iterations, is_visualize, "train", dir_path_id_img_train)

    optimizer.zero_grad()
    loss_train.backward()
    # prevent potential exploding gradients
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()
    scheduler.step()

    if nb_actual_iterations % TENSORBOARD_UPDATE == 0 or j == N_ITERATIONS - 1 :

        with torch.no_grad() :

            [loss_val, metrics_val] = compute_loss(model, X_state_val, nb_actual_iterations, is_visualize, "val", dir_path_id_img_val)
            dloss_train_dW = compute_dloss_dW(model)
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

with torch.no_grad() :

    [loss_test, metrics_test] = compute_loss(model, X_state_test, nb_actual_iterations, is_visualize, "test", dir_path_id_img_test)

    tb_writer.add_scalar('Test Loss', loss_test.detach().cpu(), nb_actual_iterations)
    tb_writer.add_scalar('Mean Test Terminal Position Distance [m]', metrics_test[0].detach().cpu(), nb_actual_iterations)
    tb_writer.add_scalar('Stddev Test Terminal Position Distance [m]', metrics_test[1].detach().cpu(), nb_actual_iterations)
    tb_writer.add_scalar('Max Test Terminal Position Distance [m]', metrics_test[2].detach().cpu(), nb_actual_iterations)

print("\nTraining Process Completed.\n")

save_model(model, nb_actual_iterations, dir_path_id_model, nn_model_state_dict_only_str, nn_model_full_str)

print("\nAll Done!\n")

sys.stdout = sys_stdout_original
file_handle_logger.close()