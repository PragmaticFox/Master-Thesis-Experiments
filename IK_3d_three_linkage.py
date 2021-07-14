#!/bin/python3

import os
import math
import torch
import shutil
import pathlib
import numpy as np

import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# fixes a possible "Fail to allocate bitmap" issue
# https://github.com/matplotlib/mplfinance/issues/386#issuecomment-869950969
matplotlib.use("Agg")

# local import
import helper

np.random.seed(21)
torch.manual_seed(21)
# only works with newer PyTorch versions
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)

IS_UR5_ROBOT = True
IS_UR5_FK_CHECK = False

identifier_string = "IK_3d_threelinkage_"

if IS_UR5_ROBOT :

    identifier_string = "IK_3d_UR5_"

string_title_joint_angles_plot = f'\nJoint Angles in Degrees\n3D Three-Linkage Robot Inverse Kinematics\n'

string_title_terminal_energy_plot = f'\nTerminal Energy Landscape in Meters\n3D Three-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_plot = f'\nJacobian Frobenius Norm Landscape\n3D Three-Linkage Robot Inverse Kinematics\n'

string_title_terminal_energy_histogram = f'\nTerminal Energy Histogram\n3D Three-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_histogram = f'\nJacobian Frobenius Norm Histogram\n3D Three-Linkage Robot Inverse Kinematics\n'

if IS_UR5_ROBOT :

    string_title_joint_angles_plot = f'\nJoint Angles in Degrees\n3D UR5 Robot Inverse Kinematics\n'

    string_title_terminal_energy_plot = f'\nTerminal Energy Landscape in Meters\n3D UR5 Robot Inverse Kinematics\n'
    string_title_jacobian_plot = f'\nJacobian Frobenius Norm Landscape\n3D UR5 Robot Inverse Kinematics\n'

    string_title_terminal_energy_histogram = f'\nTerminal Energy Histogram\n3D UR5 Robot Inverse Kinematics\n'
    string_title_jacobian_histogram = f'\nJacobian Frobenius Norm Histogram\n3D UR5 Robot Inverse Kinematics\n'

N_DIM_THETA = 3

if IS_UR5_ROBOT:

    N_DIM_THETA = 6

N_DIM_JOINTS = N_DIM_THETA

N_DIM_X = 3

N_TRAJOPT = 1

N_DIM_X_STATE = 1*N_DIM_X

FK_ORIGIN = [0.0, 0.0, 0.0]

SAMPLE_CIRCLE = True

RADIUS_INNER = 0.0
RADIUS_OUTER = 1.0

LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

N_SLICES = 5.0

if IS_UR5_ROBOT :

    # See https://www.universal-robots.com/products/ur5-robot/
    RADIUS_INNER = 0.0
    RADIUS_OUTER = 0.85

    LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

LIMITS_PLOTS = LIMITS


if IS_UR5_ROBOT :

    LIMITS_PLOTS = LIMITS

CONSTRAINTS = [[0.0, 2.0*math.pi]] * N_DIM_THETA

if IS_UR5_ROBOT :

    CONSTRAINTS = [[0.0, 2.0*math.pi]] * N_DIM_THETA

''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


def save_script(directory):

    # saves a copy of the current python script into the folder
    shutil.copy(__file__, pathlib.Path(directory, os.path.basename(__file__)))


def dh_matrix(theta, a, d, alpha):

    device = theta.device
    n = theta.shape[0]

    transform = torch.reshape(torch.eye(4, 4), shape=(1, 4, 4)).repeat(n, 1, 1).to(device)

    transform[:, 0, 0] = torch.cos(theta)
    transform[:, 0, 1] = - torch.sin(theta) * torch.cos(alpha)
    transform[:, 0, 2] = torch.sin(theta) * torch.sin(alpha)
    transform[:, 0, 3] = a * torch.cos(theta)

    transform[:, 1, 0] = torch.sin(theta)
    transform[:, 1, 1] = torch.cos(theta) * torch.cos(alpha)
    transform[:, 1, 2] = - torch.cos(theta) * torch.sin(alpha)
    transform[:, 1, 3] = a * torch.sin(theta)

    transform[:, 2, 1] = torch.sin(alpha)
    transform[:, 2, 2] = torch.cos(alpha)
    transform[:, 2, 3] = d

    return transform


def fk(theta):

    # we use homogeneous coordinates

    device = theta.device
    n_batch_times_n_trajOpt = theta.shape[0]

    p = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    p_final = torch.reshape(torch.tensor([0.0, 0.0, 0.0, 1.0]), shape=(1, 1, 4)).repeat(n_batch_times_n_trajOpt, N_DIM_JOINTS, 1).to(device)

    if IS_UR5_ROBOT:

        # UR5 robot DH matrix parameters

        a_0 = 0.0
        a_1 = -0.425
        a_2 = -0.39225
        a_3 = 0.0
        a_4 = 0.0
        a_5 = 0.0

        d_0 = 0.089159
        d_1 = 0.0
        d_2 = 0.0
        d_3 = 0.10915
        d_4 = 0.09465
        d_5 = 0.0823

        alpha_0 = math.pi/2.0
        alpha_1 = 0.0
        alpha_2 = 0.0
        alpha_3 = math.pi/2.0
        alpha_4 = -math.pi/2.0
        alpha_5 = 0.0

        torch_zeros = torch.zeros(n_batch_times_n_trajOpt).to(device)

        alpha_0 = alpha_0 + torch_zeros
        alpha_1 = alpha_1 + torch_zeros
        alpha_2 = alpha_2 + torch_zeros
        alpha_3 = alpha_3 + torch_zeros
        alpha_4 = alpha_4 + torch_zeros
        alpha_5 = alpha_5 + torch_zeros

        if IS_UR5_FK_CHECK :

            assert(theta.shape[0] >= 3, "Theta.shape[0] must be > = 3.")

            theta[0, 0] = 0.0
            theta[0, 1] = 0.0
            theta[0, 2] = 0.0
            theta[0, 3] = 0.0
            theta[0, 4] = 0.0
            theta[0, 5] = 0.0

            theta[1, 0] = 1.658062789
            theta[1, 1] = 5.235987756
            theta[1, 2] = 2.844886681
            theta[1, 3] = 4.01425728
            theta[1, 4] = 2.00712864
            theta[1, 5] = 0.523598776

            theta[2, 0] = 0.0
            theta[2, 1] = 1.623156204
            theta[2, 2] = 2.862339973
            theta[2, 3] = 5.585053606
            theta[2, 4] = 4.36332313
            theta[2, 5] = 2.862339973

            # Reference values are made with the Excel tool from here
            # https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

            p_reference = torch.tensor(
                [
                    [-0.81725, -0.19145, -0.00549, 1.0],
                    [0.09445, -0.22632, 0.02455, 1.0],
                    [-0.00825, -0.08100, 0.07599, 1.0],

                ],
                dtype = torch.float64
            ).to(device)

        dh_matrix_0 = dh_matrix(theta[:, 0], a_0, d_0, alpha_0).clone()
        dh_matrix_1 = dh_matrix(theta[:, 1], a_1, d_1, alpha_1).clone()
        dh_matrix_2 = dh_matrix(theta[:, 2], a_2, d_2, alpha_2).clone()
        dh_matrix_3 = dh_matrix(theta[:, 3], a_3, d_3, alpha_3).clone()
        dh_matrix_4 = dh_matrix(theta[:, 4], a_4, d_4, alpha_4).clone()
        dh_matrix_5 = dh_matrix(theta[:, 5], a_5, d_5, alpha_5).clone()

        transform = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, N_DIM_JOINTS+1, 1, 1).to(device)

        transform[:, 1] = torch.matmul(transform[:, 0].clone(), dh_matrix_0)
        transform[:, 2] = torch.matmul(transform[:, 1].clone(), dh_matrix_1)
        transform[:, 3] = torch.matmul(transform[:, 2].clone(), dh_matrix_2)
        transform[:, 4] = torch.matmul(transform[:, 3].clone(), dh_matrix_3)
        transform[:, 5] = torch.matmul(transform[:, 4].clone(), dh_matrix_4)
        transform[:, 6] = torch.matmul(transform[:, 5].clone(), dh_matrix_5)

        for i in range(N_DIM_JOINTS):

            p_final[:, i] = torch.matmul(torch.clone(transform[:, i+1]), p)

        if IS_UR5_FK_CHECK :

            diff = p_reference - p_final[0:3, -1]
            normed_diff = torch.norm(diff)
            print(f"\nChecking UR5 FK\nError Vector Norm: {normed_diff}\n")

            exit(0)

        return p_final[:, :, :-1]

    rt_hom = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, N_DIM_JOINTS+1, 1, 1).to(device)
    r_hom_i = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, N_DIM_JOINTS+1, 1, 1).to(device)
    t_hom_i = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, N_DIM_JOINTS+1, 1, 1).to(device)

    for i in range(N_DIM_THETA):

        if i % 3 == 0 :

            # rotation around x-axis (yz-plane)

            t_hom_i[:, i, 1, 3] = 1.0

            r_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])
            r_hom_i[:, i, 1, 2] = -torch.sin(theta[:, i])
            r_hom_i[:, i, 2, 1] = torch.sin(theta[:, i])
            r_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])

        if i % 3 == 1 :

            # rotation around z-axis (xy-plane)

            t_hom_i[:, i, 0, 3] = 1.0

            r_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            r_hom_i[:, i, 0, 1] = -torch.sin(theta[:, i])
            r_hom_i[:, i, 1, 0] = torch.sin(theta[:, i])
            r_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])

        if i % 3 == 2:

            # rotation around z-axis (xy-plane)

            t_hom_i[:, i, 0, 3] = 1.0

            r_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            r_hom_i[:, i, 0, 1] = -torch.sin(theta[:, i])
            r_hom_i[:, i, 1, 0] = torch.sin(theta[:, i])
            r_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])

        tmp = torch.matmul(torch.clone(rt_hom[:, i]), torch.clone(r_hom_i[:, i]))
        rt_hom[:, i+1] = torch.matmul(torch.clone(tmp),torch.clone(t_hom_i[:, i]))
        p_final[:, i] = torch.matmul(rt_hom[:, i+1], p)

    return p_final[:, :, :-1]


def visualize_trajectory_and_save_image(x_state, x_hat_fk_chain, dir_path_img, fname_img):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_state[0], x_state[1], x_state[2], c='r', s=100, zorder=-10)

    for t in range(N_TRAJOPT):

        ax.quiver(
            FK_ORIGIN[0],
            FK_ORIGIN[1],
            FK_ORIGIN[2],
            x_hat_fk_chain[0, t, 0] - FK_ORIGIN[0],
            x_hat_fk_chain[0, t, 1] - FK_ORIGIN[1],
            x_hat_fk_chain[0, t, 2] - FK_ORIGIN[2],
            color='b',
            alpha=0.8,
            normalize=False
        )

        for i in range(1, N_DIM_JOINTS, 1):

            ax.scatter(x_hat_fk_chain[i-1, t, 0], x_hat_fk_chain[i-1,
                       t, 1], x_hat_fk_chain[i-1, t, 2], c='0.5', s=10)

            ax.quiver(
                x_hat_fk_chain[i-1, t, 0],
                x_hat_fk_chain[i-1, t, 1],
                x_hat_fk_chain[i-1, t, 2],
                x_hat_fk_chain[i, t, 0] - x_hat_fk_chain[i-1, t, 0],
                x_hat_fk_chain[i, t, 1] - x_hat_fk_chain[i-1, t, 1],
                x_hat_fk_chain[i, t, 2] - x_hat_fk_chain[i-1, t, 2],
                color='b',
                alpha=0.8,
                normalize=False
            )

        ax.scatter(x_hat_fk_chain[-1, t, 0], x_hat_fk_chain[-1,
                   t, 1], x_hat_fk_chain[-1, t, 2], c='k', s=10)

    ax.scatter(FK_ORIGIN[0], FK_ORIGIN[1], FK_ORIGIN[2], c='0.5', s=10)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.gca().set_aspect('auto', adjustable='box')

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    # close the plot handle
    plt.close("all")


def compute_and_save_samples_plot(X_state_train, X_state_val, X_state_test, dir_path_img, fname_img):

    ax = plt.axes(projection='3d')

    ax.plot(X_state_train[:, 0], X_state_train[:, 1],
            X_state_train[:, 2], ms=1.0, marker='o', color='b', ls='')
    ax.plot(X_state_val[:, 0], X_state_val[:, 1],
            X_state_val[:, 2], ms=1.0, marker='o', color='g', ls='')
    ax.plot(X_state_test[:, 0], X_state_test[:, 1],
            X_state_test[:, 2], ms=1.0, marker='o', color='r', ls='')

    plt.gca().set_aspect('auto', adjustable='box')

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')


def compute_energy(model, x_state, is_constrained):

    n_batch = x_state.shape[0]

    theta_hat = model(x_state)

    x_hat_fk_chain = fk(theta_hat)

    x_hat_fk_chain = torch.reshape(input=x_hat_fk_chain, shape=(
        n_batch, N_TRAJOPT, N_DIM_JOINTS, N_DIM_X_STATE))
    x_hat_fk_chain = torch.transpose(input=x_hat_fk_chain, dim0=1, dim1=2)

    x_hat_state = x_hat_fk_chain[:, -1, -1, :]

    terminal_position_distance = torch.norm(
        (x_state - x_hat_state), p=2, dim=-1)
    energy = torch.pow(terminal_position_distance, exponent=2)

    constraint_bound = torch.zeros_like(energy)

    if is_constrained:
        constraint_bound = helper.soft_bound_constraint(
            lower_limit=-math.pi, upper_limit=0.0, eps_rel=1e-1, stiffness=1e-0, x=theta_hat[:, -1])
        #constraint_bound = helper.soft_bound_constraint(lower_limit = 0.0, upper_limit = math.pi, eps_rel = 1e-1, stiffness = 1e-0, x = theta_hat[:, -1])

    energy += constraint_bound

    return energy, constraint_bound, terminal_position_distance, x_hat_fk_chain


def compute_and_save_joint_angles_plot(rng, model, device, X_state_train, dpi, n_one_dim, dir_path_img, fname_img, fontdict, title_string):

    delta_z = (LIMITS_PLOTS[2][1] - LIMITS_PLOTS[2][0]) / N_SLICES

    for i in range(int(N_SLICES)) :

        xs = X_state_train[:, 0].detach().cpu()
        ys = X_state_train[:, 1].detach().cpu()
        zs = X_state_train[:, 2].detach().cpu()

        z_min = LIMITS_PLOTS[2][0] + i*delta_z
        z_max = LIMITS_PLOTS[2][0] + (i+1)*delta_z

        indices_max = zs <= z_max

        xs_ = xs[indices_max]
        ys_ = ys[indices_max]
        zs_ = zs[indices_max]

        indices_min = zs_ >= z_min

        xs__ = xs_[indices_min]
        ys__ = ys_[indices_min]

        alpha_train_samples = 0.25

        dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
        dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

        dimX, dimY = np.meshgrid(dimX, dimY)

        dimZ = np.array([z_min + rng.uniform(0, 1) * ( z_max - z_min ) for _ in range(len(dimX.flatten()))])

        x_state = torch.tensor(
            np.stack((dimX.flatten(), dimY.flatten(), dimZ.flatten()), axis=-1)).to(device)

        theta_hat = torch.zeros((n_one_dim*n_one_dim, N_TRAJOPT, N_DIM_THETA))

        with torch.no_grad():

            if n_one_dim > 100:

                n_splits = 100

                delta = n_one_dim*n_one_dim // n_splits

                for split in range(n_splits):
                    theta_hat_tmp = model(x_state[split*delta:(split+1)*delta])
                    # print(theta_hat_tmp.shape)
                    # print(theta_hat[split*delta:(split+1)*delta].shape)
                    #print(torch.reshape(theta_hat_tmp, (delta, N_TRAJOPT, N_DIM_THETA)).shape)
                    theta_hat[split*delta:(split+1)*delta] = torch.reshape(
                        theta_hat_tmp, (delta, N_TRAJOPT, N_DIM_THETA))

            else:

                theta_hat = model(x_state)

        theta_hat = (theta_hat % (2.0 * math.pi)) * 180.0 / math.pi

        theta_hat = torch.reshape(input=theta_hat, shape=(
            n_one_dim, n_one_dim, N_TRAJOPT, N_DIM_THETA)).detach().cpu()

        rad_min = 0.0
        rad_max = 360.0

        c = 0

        for j in range(N_DIM_THETA):

            # plot

            fig, ax = plt.subplots()

            plt.subplots_adjust(left=0, bottom=0, right=1.25,
                                top=1.25, wspace=1, hspace=1)

            ax.set_aspect(aspect='equal', adjustable='box')

            ax.set_title(
                f'\nJoint {j+1}\n' + f"\nz = [{z_min}, {z_max}]\n" + title_string,
                fontdict=fontdict,
                pad=5
            )

            ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
            c = ax.pcolormesh(dimX, dimY, theta_hat[:, :, -1, j], cmap='RdYlBu', shading='gouraud', vmin=rad_min, vmax=rad_max)

            ax.plot(xs__, ys__, ms=helper.TRAIN_SAMPLE_POINTS_PLOT_SIZE_3D, marker='o', color='k', ls='', alpha=alpha_train_samples)

            cb = fig.colorbar(c, ax=ax, extend='max')

            plt.xlabel("x")
            plt.ylabel("y")

            helper.save_figure(fig, dpi, dir_path_img, str(i+1) + "_" + str(j+1) + "_" + fname_img)

            # close the plot handle
            plt.close('all')


def compute_and_save_jacobian_plot(rng, model, device, X_state_train, dpi, n_one_dim, dir_path_img, fname_img, fontdict, title_string):

    n_samples = n_one_dim*n_one_dim

    x_state = torch.tensor([helper.compute_sample(rng, LIMITS, SAMPLE_CIRCLE, RADIUS_OUTER, RADIUS_INNER) for _ in range(n_samples)], dtype=helper.DTYPE_TORCH).to(device)

    model_sum = lambda x : torch.sum(model(x), axis = 0)
    jac = torch.zeros(size=(n_samples, N_TRAJOPT*N_DIM_THETA, N_DIM_X))

    if n_one_dim > 100 :

        n_splits = 100

        delta = n_samples // n_splits

        for split in range(n_splits) :

            jac[split*delta:(split+1)*delta] = torch.autograd.functional.jacobian(model_sum, x_state[split*delta:(split+1)*delta], create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

    else :

        jac = torch.autograd.functional.jacobian(model_sum, x_state, create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

    jac_norm = torch.norm(jac.reshape(n_samples, N_TRAJOPT*N_DIM_THETA*N_DIM_X), p="fro", dim=-1)
    jac_norm = np.array(jac_norm.detach().cpu().tolist())

    dimX = x_state[:, 0].detach().cpu()
    dimY = x_state[:, 1].detach().cpu()
    dimZ = x_state[:, 2].detach().cpu()

    # plot

    ax = plt.axes(projection='3d')

    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=1, hspace=1)

    ax.set_aspect(aspect='auto', adjustable='box')

    ax.set_title(
        title_string,
        fontdict=fontdict,
        pad=5
    )

    cmap = pl.cm.RdBu
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.logspace(helper.ALPHA_PARAM_3D_PLOTS, 0, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    c = ax.scatter(
        xs=dimX,
        ys=dimY,
        zs=dimZ,
        zdir='z',
        s=20,
        c=jac_norm,
        depthshade=True,
        cmap=my_cmap,
        norm=matplotlib.colors.LogNorm(
            vmin=helper.COLORBAR_JACOBIAN_LOWER_THRESHOLD, vmax=helper.COLORBAR_JACOBIAN_UPPER_THRESHOLD)
    )

    ax.set_xlim(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1])
    ax.set_ylim(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1])
    ax.set_zlim(LIMITS_PLOTS[2][0], LIMITS_PLOTS[2][1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig = plt.gcf()

    cb = fig.colorbar(plt.cm.ScalarMappable(norm = matplotlib.colors.LogNorm(
            vmin=helper.COLORBAR_JACOBIAN_LOWER_THRESHOLD, vmax=helper.COLORBAR_JACOBIAN_UPPER_THRESHOLD), cmap = "RdBu"), ax = ax, extend = "max")

    helper.save_figure(fig, dpi, dir_path_img, fname_img)
    # close the plot handle
    plt.close('all')

    delta_z = (LIMITS_PLOTS[2][1] - LIMITS_PLOTS[2][0]) / N_SLICES

    for i in range(int(N_SLICES)) :

        xs = X_state_train[:, 0].detach().cpu()
        ys = X_state_train[:, 1].detach().cpu()
        zs = X_state_train[:, 2].detach().cpu()

        z_min = LIMITS_PLOTS[2][0] + i*delta_z
        z_max = LIMITS_PLOTS[2][0] + (i+1)*delta_z

        indices_max = zs <= z_max

        xs_ = xs[indices_max]
        ys_ = ys[indices_max]
        zs_ = zs[indices_max]

        indices_min = zs_ >= z_min

        xs__ = xs_[indices_min]
        ys__ = ys_[indices_min]

        alpha_train_samples = 0.25

        dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
        dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

        dimX, dimY = np.meshgrid(dimX, dimY)

        dimZ = np.array([z_min + rng.uniform(0, 1) * ( z_max - z_min ) for _ in range(len(dimX.flatten()))])

        x_state = torch.tensor(
            np.stack((dimX.flatten(), dimY.flatten(), dimZ.flatten()), axis=-1)).to(device)

        model_sum = lambda x : torch.sum(model(x), axis = 0)

        jac = torch.zeros(size=(n_one_dim*n_one_dim, N_TRAJOPT*N_DIM_THETA, N_DIM_X))

        if n_one_dim > 100 :

            n_splits = 100

            delta = n_one_dim*n_one_dim // n_splits

            for split in range(n_splits) :

                jac[split*delta:(split+1)*delta] = torch.autograd.functional.jacobian(model_sum, x_state[split*delta:(split+1)*delta], create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

        else :

            jac = torch.autograd.functional.jacobian(model_sum, x_state, create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

        jac_norm = torch.reshape(jac, shape=(
            n_one_dim, n_one_dim, N_TRAJOPT*N_DIM_THETA*N_DIM_X))
        jac_norm = torch.norm(jac_norm, p="fro", dim=-1)

        jac_norm = np.array(jac_norm.detach().cpu().reshape(
            (n_one_dim, n_one_dim)).tolist())
        jac_norm_min = jac_norm.min()
        jac_norm_max = jac_norm.max()

        # plot

        fig, ax = plt.subplots()

        plt.subplots_adjust(left=0, bottom=0, right=1.25,
                            top=1.25, wspace=1, hspace=1)

        ax.set_aspect(aspect='equal', adjustable='box')

        ax.set_title(
            f"\nz = [{z_min}, {z_max}]\n" + title_string,
            fontdict=fontdict,
            pad=5
        )

        ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
        c = ax.pcolormesh(dimX, dimY, jac_norm, cmap='RdBu', shading='gouraud',
                        norm=matplotlib.colors.LogNorm(vmin=helper.COLORBAR_JACOBIAN_LOWER_THRESHOLD, vmax=helper.COLORBAR_JACOBIAN_UPPER_THRESHOLD))

        ax.plot(xs__, ys__, ms=helper.TRAIN_SAMPLE_POINTS_PLOT_SIZE_3D,
                marker='o', color='k', ls='', alpha=alpha_train_samples)

        cb = fig.colorbar(c, ax=ax, extend='max')

        plt.xlabel("x")
        plt.ylabel("y")

        helper.save_figure(fig, dpi, dir_path_img, str(i+1) + "_" + fname_img)

        # close the plot handle
        plt.close('all')


def compute_and_save_terminal_energy_plot(rng, model, device, X_state_train, dpi, is_constrained, n_one_dim, dir_path_img, fname_img, fontdict, title_string):
    
    n_samples = n_one_dim*n_one_dim

    x_state = torch.tensor([helper.compute_sample(rng, LIMITS, SAMPLE_CIRCLE, RADIUS_OUTER, RADIUS_INNER)
                           for _ in range(n_samples)], dtype=helper.DTYPE_TORCH).to(device)

    terminal_energy = torch.zeros((n_samples)).to(device)

    with torch.no_grad():

        if n_one_dim > 100:

            n_splits = 100

            delta = n_one_dim*n_one_dim // n_splits

            for split in range(n_splits):
                energy_tmp, constraint_tmp, terminal_position_distance_tmp, _ = compute_energy(
                    model, x_state[split*delta:(split+1)*delta], is_constrained)
                # energy_tmp
                terminal_energy[split*delta:(split+1) *
                                delta] = terminal_position_distance_tmp

        else:

            energy, constraint, terminal_position_distance, _ = compute_energy(
                model, x_state, is_constrained)
            terminal_energy = terminal_position_distance

    terminal_energy = terminal_energy.detach().cpu()

    dimX = x_state[:, 0].detach().cpu()
    dimY = x_state[:, 1].detach().cpu()
    dimZ = x_state[:, 2].detach().cpu()

    # plot

    ax = plt.axes(projection='3d')

    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=1, hspace=1)

    ax.set_aspect(aspect='auto', adjustable='box')

    ax.set_title(
        title_string,
        fontdict=fontdict,
        pad=5
    )

    cmap = pl.cm.RdBu
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.logspace(helper.ALPHA_PARAM_3D_PLOTS, 0, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    c = ax.scatter(
        xs=dimX,
        ys=dimY,
        zs=dimZ,
        zdir='z',
        s=20,
        c=terminal_energy,
        depthshade=True,
        cmap=my_cmap,
        norm=matplotlib.colors.LogNorm(
            vmin=helper.COLORBAR_ENERGY_LOWER_THRESHOLD, vmax=helper.COLORBAR_ENERGY_UPPER_THRESHOLD)
    )

    ax.set_xlim(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1])
    ax.set_ylim(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1])
    ax.set_zlim(LIMITS_PLOTS[2][0], LIMITS_PLOTS[2][1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig = plt.gcf()

    cb = fig.colorbar(plt.cm.ScalarMappable(norm = matplotlib.colors.LogNorm(
            vmin=helper.COLORBAR_ENERGY_LOWER_THRESHOLD, vmax=helper.COLORBAR_ENERGY_UPPER_THRESHOLD), cmap = "RdBu"), ax = ax, extend = "max")

    helper.save_figure(fig, dpi, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')

    delta_z = (LIMITS_PLOTS[2][1] - LIMITS_PLOTS[2][0]) / N_SLICES

    for i in range(int(N_SLICES)):

        xs = X_state_train[:, 0].detach().cpu()
        ys = X_state_train[:, 1].detach().cpu()
        zs = X_state_train[:, 2].detach().cpu()

        z_min = LIMITS_PLOTS[2][0] + i*delta_z
        z_max = LIMITS_PLOTS[2][0] + (i+1)*delta_z

        indices_max = zs <= z_max

        xs_ = xs[indices_max]
        ys_ = ys[indices_max]
        zs_ = zs[indices_max]

        indices_min = zs_ >= z_min

        xs__ = xs_[indices_min]
        ys__ = ys_[indices_min]

        alpha_train_samples = 0.25

        dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
        dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

        dimX, dimY = np.meshgrid(dimX, dimY)

        dimZ = np.array([z_min + rng.uniform(0, 1) * ( z_max - z_min ) for _ in range(len(dimX.flatten()))])

        x_state = torch.tensor(
            np.stack((dimX.flatten(), dimY.flatten(), dimZ.flatten()), axis=-1)).to(device)

        terminal_energy = torch.zeros((x_state.shape[0])).to(device)

        with torch.no_grad():

            if n_one_dim > 100:

                n_splits = 100

                delta = n_one_dim*n_one_dim // n_splits

                for split in range(n_splits):
                    energy_tmp, constraint_tmp, terminal_position_distance_tmp, _ = compute_energy(
                        model, x_state[split*delta:(split+1)*delta], is_constrained)
                    # energy_tmp
                    terminal_energy[split*delta:(split+1) *
                                    delta] = terminal_position_distance_tmp

            else:

                energy, constraint, terminal_position_distance, _ = compute_energy(
                    model, x_state, is_constrained)
                terminal_energy = terminal_position_distance

        terminal_energy = np.array(terminal_energy.detach(
        ).cpu().reshape((n_one_dim, n_one_dim)).tolist())
        terminal_energy_min = terminal_energy.min()
        terminal_energy_max = terminal_energy.max()

        # plot

        fig, ax = plt.subplots()

        plt.subplots_adjust(left=0, bottom=0, right=1.25,
                            top=1.25, wspace=1, hspace=1)

        ax.set_aspect(aspect='equal', adjustable='box')

        ax.set_title(
            f"\nz = [{z_min}, {z_max}]\n" + title_string,
            fontdict=fontdict,
            pad=5
        )

        ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
        c = ax.pcolormesh(dimX, dimY, terminal_energy, cmap='RdBu', shading='gouraud',
                        norm=matplotlib.colors.LogNorm(vmin=helper.COLORBAR_ENERGY_LOWER_THRESHOLD, vmax=helper.COLORBAR_ENERGY_UPPER_THRESHOLD))

        ax.plot(xs__, ys__, ms=helper.TRAIN_SAMPLE_POINTS_PLOT_SIZE_3D,
                marker='o', color='k', ls='', alpha=alpha_train_samples)

        cb = fig.colorbar(c, ax=ax, extend='max')

        plt.xlabel("x")
        plt.ylabel("y")

        helper.save_figure(fig, dpi, dir_path_img, str(i+1) + "_" + fname_img)

        # close the plot handle
        plt.close('all')


def compute_and_save_jacobian_histogram(rng, model, X_samples, dpi, dir_path_img, fname_img, fontdict, title_string):

    n_samples = X_samples.shape[0]

    model_sum = lambda x : torch.sum(model(x), axis = 0)

    jac = torch.zeros(size=(n_samples, N_TRAJOPT * N_DIM_THETA, N_DIM_X)).to(X_samples.device)
    jac = torch.autograd.functional.jacobian(model_sum, X_samples, create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

    jac = jac.reshape(n_samples, N_TRAJOPT * N_DIM_THETA * N_DIM_X)

    jac_norm = torch.norm(jac, p="fro", dim=-1)
    jac_norm = np.array(jac_norm.detach().cpu().tolist())

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=1, hspace=1)

    ax.set_title(
        title_string,
        fontdict=fontdict,
        pad=5
    )

    arr = jac_norm.flatten()

    helper.plot_histogram(plt, ax, arr)

    helper.save_figure(fig, dpi, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_terminal_energy_histogram(rng, model, X_samples, dpi, is_constrained, dir_path_img, fname_img, fontdict, title_string):

    n_samples = X_samples.shape[0]

    terminal_energy = torch.zeros((n_samples)).to(X_samples.device)

    energy, constraint, terminal_position_distance, _ = compute_energy(
        model, X_samples, is_constrained)

    terminal_energy = np.array(
        terminal_position_distance.detach().cpu().tolist())

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=1, hspace=1)

    ax.set_title(
        title_string,
        fontdict=fontdict,
        pad=5
    )

    arr = terminal_energy.flatten()

    helper.plot_histogram(plt, ax, arr)

    helper.save_figure(fig, dpi, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')


def plot_joint_angles_region_slices(plt, dpi, dir_path_img, fname_img, delta, case, i, xs, xs_min, xs_max, ys, ys_min, ys_max, zs, zs_min, zs_max):

    ax = plt.axes()

    indices_max = 0

    if case == "x" :

        indices_max = xs <= xs_min + (i+1)*delta

    if case == "y" :

        indices_max = ys <= ys_min + (i+1)*delta

    if case == "z" :

        indices_max = zs <= zs_min + (i+1)*delta

    xs_ = xs[indices_max]
    ys_ = ys[indices_max]
    zs_ = zs[indices_max]

    indices_min = 0

    if case == "x" :

        indices_min = xs_ >= xs_min + i*delta

    if case == "y" :

        indices_min = ys_ >= ys_min + i*delta

    if case == "z" :

        indices_min = zs_ >= zs_min + i*delta

    xs__ = xs_[indices_min]
    ys__ = ys_[indices_min]
    zs__ = zs_[indices_min]

    xs_min__ = xs__.min()
    xs_max__ = xs__.max()

    ys_min__ = ys__.min()
    ys_max__ = ys__.max()

    zs_min__ = zs__.min()
    zs_max__ = zs__.max()

    x_min__ = min(xs_min__, LIMITS[0][0])
    x_max__ = max(xs_max__, LIMITS[0][1])

    y_min__ = min(ys_min__, LIMITS[1][0])
    y_max__ = max(ys_max__, LIMITS[1][1])

    z_min__ = min(zs_min__, LIMITS[2][0])
    z_max__ = max(zs_max__, LIMITS[2][1])

    if case == "x" :

        ax.plot(ys__, zs__, ms=1.0, marker='o', color='b', ls='', alpha=0.5)

        ax.set_xlim(y_min__, y_max__)
        ax.set_ylim(z_min__, z_max__)

        ax.set_xlabel("y")
        ax.set_ylabel("z")

    if case == "y" :

        ax.plot(xs__, zs__, ms=1.0, marker='o', color='b', ls='', alpha=0.5)

        ax.set_xlim(x_min__, x_max__)
        ax.set_ylim(z_min__, z_max__)

        ax.set_xlabel("x")
        ax.set_ylabel("z")

    if case == "z" :

        ax.plot(xs__, ys__, ms=1.0, marker='o', color='b', ls='', alpha=0.5)

        ax.set_xlim(x_min__, x_max__)
        ax.set_ylim(y_min__, y_max__)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.set_title(
        f"\nx = [{xs_min__}, {xs_max__}]\ny = [{ys_min__}, {ys_max__}]\nz = [{zs_min__}, {zs_max__}]\n",
        # fontdict=fontdict,
        pad=5
    )

    ax.set_aspect('auto', adjustable='box')

    fig = plt.gcf()

    helper.save_figure(fig, dpi, dir_path_img, f"{case}-axis_slices_" + str(i+1) + "_" + fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_joint_angles_region_plot(rng, device, n_samples_theta, dpi, dir_path_img, fname_img):

    theta = torch.tensor([helper.sample_joint_angles(rng, CONSTRAINTS) for _ in range(
        n_samples_theta)], dtype=helper.DTYPE_TORCH).to(device)

    x_fk_chain = torch.zeros(size=(n_samples_theta, N_TRAJOPT*N_DIM_JOINTS, N_DIM_X_STATE))

    if n_samples_theta > 1000:

        n_splits = 1000

        delta = n_samples_theta // n_splits

        for split in range(n_splits):

            x_fk_chain[split*delta:(split+1)*delta] = fk(theta[split*delta:(split+1)*delta])

    else:

        x_fk_chain = fk(theta)

    x_fk_chain = torch.reshape(input=x_fk_chain, shape=(
        n_samples_theta, N_TRAJOPT, N_DIM_JOINTS, N_DIM_X_STATE))
    x_fk_chain = torch.transpose(input=x_fk_chain, dim0=1, dim1=2)
    x_fk_chain = x_fk_chain.detach().cpu()

    xs = x_fk_chain[:, -1, -1, 0]
    ys = x_fk_chain[:, -1, -1, 1]
    zs = x_fk_chain[:, -1, -1, 2]

    xs_min = xs.min()
    xs_max = xs.max()

    ys_min = ys.min()
    ys_max = ys.max()

    zs_min = zs.min()
    zs_max = zs.max()

    x_min = min(xs_min, LIMITS[0][0])
    x_max = max(xs_max, LIMITS[0][1])

    y_min = min(ys_min, LIMITS[1][0])
    y_max = max(ys_max, LIMITS[1][1])

    z_min = min(zs_min, LIMITS[2][0])
    z_max = max(zs_max, LIMITS[2][1])

    '''
    ax = plt.axes(projection='3d')

    ax.plot(xs, ys, zs, ms=1.0, marker='o', color='b', ls='', alpha=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(
        f"\nx = [{xs_min}, {xs_max}]\ny = [{ys_min}, {ys_max}]\nz = [{zs_min}, {zs_max}]\n",
        # fontdict=fontdict,
        pad=5
    )

    plt.gca().set_aspect('auto', adjustable='box')

    helper.save_figure(plt.gcf(), dpi, dir_path_img,
                       identifier_string + "joint_angles_region_plot_3d.png")

    # close the plot handle
    plt.close('all')
    '''

    case_x = "x"
    case_y = "y"
    case_z = "z"

    delta_x = (xs_max - xs_min) / N_SLICES
    delta_y = (ys_max - ys_min) / N_SLICES
    delta_z = (zs_max - zs_min) / N_SLICES

    for i in range(int(N_SLICES)):
        
        plot_joint_angles_region_slices(plt, dpi, dir_path_img, fname_img, delta_x, case_x, i, xs, xs_min, xs_max, ys, ys_min, ys_max, zs, zs_min, zs_max)
        plot_joint_angles_region_slices(plt, dpi, dir_path_img, fname_img, delta_y, case_y, i, xs, xs_min, xs_max, ys, ys_min, ys_max, zs, zs_min, zs_max)
        plot_joint_angles_region_slices(plt, dpi, dir_path_img, fname_img, delta_z, case_z, i, xs, xs_min, xs_max, ys, ys_min, ys_max, zs, zs_min, zs_max)

