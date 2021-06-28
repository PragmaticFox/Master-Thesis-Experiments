#!/bin/python3

import math
import random
import pathlib

import torch
import numpy as np

import matplotlib.pyplot as plt

import helper

identifier_string = "IK_3d_"

string_title_joint_angles_plot = f'\nJoint Angles in Degrees\n3D Three-Linkage Robot Inverse Kinematics\n'

string_title_heatmap_plot = f'\nTerminal Energy Landscape in Meters\n3D Three-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_plot = f'\nJacobian Frobenius Norm Landscape\n3D Three-Linkage Robot Inverse Kinematics\n'

string_title_heatmap_histogram = f'\nTerminal Energy Histogram\n3D Three-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_histogram = f'\nJacobian Frobenius Norm Histogram\n3D Three-Linkage Robot Inverse Kinematics\n'

N_DIM_THETA = 3
N_DIM_X = 3

N_TRAJOPT = 1

N_DIM_X_STATE = 1*N_DIM_X

LR_INITIAL = 1e-2

# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99925 # for 10k
# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99975 # for 30k
LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99985  # for 50k
# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.999925 # for 100k

FK_ORIGIN = [0.0, 0.0, 0.0]

RADIUS_INNER = 0.0
RADIUS_OUTER = 1.0

SAMPLE_CIRCLE = True

LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

LIMITS_PLOTS = LIMITS
LIMITS_PLOTS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]
#LENGTHS = N_DIM_THETA*[(3.0 + 1e-3)/N_DIM_THETA]


''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


def fk(theta):

    device = theta.device
    n_batch_times_n_trajOpt = theta.shape[0]
    n_dim_theta = theta.shape[1]

    p = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    p_final = torch.reshape(torch.tensor([0.0, 0.0, 0.0, 1.0]), shape=(
        1, 1, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1).to(device)
    #rt_hom = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)
    rt_hom = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, 1, 1, 1).to(device)
    rt_hom_i = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(
        n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)

    for i in range(n_dim_theta):

        if (i % 3 == 0):

            # rotation around x-axis (yz-plane)
            # homogeneous coordinates

            #rt_hom_i[:, i, 0, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 1, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 2, 3] = LENGTHS[i]

            rt_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])
            rt_hom_i[:, i, 1, 2] = -torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 1] = torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])

        if (i % 3 == 1):

            # rotation around y-axis (xz-plane)
            # homogeneous coordinates

            #rt_hom_i[:, i, 0, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 1, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 2, 3] = LENGTHS[i]

            rt_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            rt_hom_i[:, i, 0, 2] = torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 0] = -torch.sin(theta[:, i])
            rt_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])

        if (i % 3 == 2):

            # rotation around z-axis (xy-plane)
            # homogeneous coordinates

            rt_hom_i[:, i, 0, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 1, 3] = LENGTHS[i]
            #rt_hom_i[:, i, 2, 3] = LENGTHS[i]

            rt_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            rt_hom_i[:, i, 0, 1] = -torch.sin(theta[:, i])
            rt_hom_i[:, i, 1, 0] = torch.sin(theta[:, i])
            rt_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])
        
        tmp = torch.matmul(rt_hom[:, i], rt_hom_i[:, i]).reshape(shape = (n_batch_times_n_trajOpt, 1, 4, 4))

        print(tmp.shape)
        print(rt_hom.shape)
        rt_hom = torch.cat(tensors = (rt_hom, tmp), dim = 1)
        print(rt_hom.shape)
        #rt_hom[:, i+1] = torch.matmul(rt_hom[:, i], rt_hom_i[:, i])
        p_final[:, i+1] = torch.matmul(rt_hom[:, i+1], p)

    return p_final[:, 1:, :-1]


def compute_sample():

    x = [
        LIMITS[0][0] + random.uniform(0, 1)*(LIMITS[0][1] - LIMITS[0][0]),
        LIMITS[1][0] + random.uniform(0, 1)*(LIMITS[1][1] - LIMITS[1][0]),
        LIMITS[2][0] + random.uniform(0, 1)*(LIMITS[2][1] - LIMITS[2][0])
    ]

    if SAMPLE_CIRCLE:

        if RADIUS_OUTER <= RADIUS_INNER:

            print(f"Make sure RADIUS_OUTER > RADIUS_INNER!")
            exit(1)

        r = np.linalg.norm(x, ord=2)

        while r >= RADIUS_OUTER or r < RADIUS_INNER:

            x = [
                LIMITS[0][0] +
                random.uniform(0, 1)*(LIMITS[0][1] - LIMITS[0][0]),
                LIMITS[1][0] +
                random.uniform(0, 1)*(LIMITS[1][1] - LIMITS[1][0]),
                LIMITS[2][0] +
                random.uniform(0, 1)*(LIMITS[2][1] - LIMITS[2][0])
            ]

            r = np.linalg.norm(x, ord=2)

    return x


def save_figure(figure, dpi, dir_path_img, fname_img):

    figure.savefig(
        fname=pathlib.Path(dir_path_img, fname_img),
        bbox_inches="tight",
        dpi=dpi
        #pil_kwargs = {'optimize': True, 'quality': 75}
    )


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
            normalize = True
        )

        for i in range(1, N_DIM_THETA, 1):

            ax.scatter(x_hat_fk_chain[i-1, t, 0], x_hat_fk_chain[i-1, t, 1], x_hat_fk_chain[i-1, t, 2], c='0.5', s=10)

            ax.quiver(
                x_hat_fk_chain[i-1, t, 0],
                x_hat_fk_chain[i-1, t, 1],
                x_hat_fk_chain[i-1, t, 2],
                x_hat_fk_chain[i, t, 0] - x_hat_fk_chain[i-1, t, 0],
                x_hat_fk_chain[i, t, 1] - x_hat_fk_chain[i-1, t, 1],
                x_hat_fk_chain[i, t, 2] - x_hat_fk_chain[i-1, t, 2],
                color='b',
                alpha=0.8,
                normalize = True
            )

    ax.scatter(x_hat_fk_chain[-1, t, 0], x_hat_fk_chain[-1, t, 1], x_hat_fk_chain[-1, t, 2], c='k', s=10)

    ax.scatter(FK_ORIGIN[0], FK_ORIGIN[1], FK_ORIGIN[2], c='0.5', s=10)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    plt.gca().set_aspect('auto', adjustable='box')

    save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close()


def compute_and_save_samples_plot(X_state_train, X_state_val, X_state_test, dir_path_img, fname_img):

    ax = plt.axes(projection='3d')

    ax.plot(X_state_train[:, 0], X_state_train[:, 1],
            X_state_train[:, 2], ms=1.0, marker='o', color='b', ls='')
    ax.plot(X_state_val[:, 0], X_state_val[:, 1],
            X_state_val[:, 2], ms=1.0, marker='o', color='g', ls='')
    ax.plot(X_state_test[:, 0], X_state_test[:, 1],
            X_state_test[:, 2], ms=1.0, marker='o', color='r', ls='')

    plt.gca().set_aspect('auto', adjustable='box')

    save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close('all')


def compute_energy(model, x_state, is_constrained):

    n_batch = x_state.shape[0]

    theta_hat = model(x_state)

    x_hat_fk_chain = fk(theta_hat)

    x_hat_fk_chain = torch.reshape(input=x_hat_fk_chain, shape=(
        n_batch, N_TRAJOPT, N_DIM_THETA, N_DIM_X_STATE))
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


def compute_and_save_joint_angles_plot(model, device, dpi, n_one_dim, dir_path_img, index, fname_img, title_string):

    return


def compute_and_save_jacobian_plot(model, device, X_state_train, dpi, n_one_dim, dir_path_img, index, fname_img, fontdict, title_string):

    return


def compute_and_save_heatmap_plot(model, device, X_state_train, metrics, dpi, is_constrained, n_one_dim, dir_path_img, index, fname_img, fontdict, title_string):

    return


def compute_and_save_jacobian_histogram(model, X_samples, dpi, dir_path_img, index, fname_img, fontdict, title_string):

    n_samples = X_samples.shape[0]

    jac = torch.zeros(size=(n_samples, N_TRAJOPT *
                            N_DIM_THETA, N_DIM_X)).to(X_samples.device)

    for i in range(n_samples):
        jac[i] = torch.reshape(torch.autograd.functional.jacobian(model, X_samples[i:i+1], create_graph=False,
                                                                  strict=False), shape=(N_TRAJOPT*N_DIM_THETA, N_DIM_X))

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

    arr = jac_norm.flatten() if len(
        jac_norm.flatten()) < 1000 else jac_norm.flatten()[:1000]

    hist, bins = np.histogram(arr, bins=25)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(x=arr, bins=logbins, density=True, log=True)
    plt.xscale('log')
    plt.grid(True)

    helper.save_figure(fig, dpi, dir_path_img, str(index) + "_" + fname_img)
    helper.save_figure(fig, dpi, "", fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_heatmap_histogram(model, X_samples, dpi, is_constrained, dir_path_img, index, fname_img, fontdict, title_string):

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

    hist, bins = np.histogram(arr, bins=25)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(x=arr, bins=logbins, density=True, log=True)
    plt.xscale('log')
    plt.grid(True)

    helper.save_figure(fig, dpi, dir_path_img, str(index) + "_" + fname_img)
    helper.save_figure(fig, dpi, "", fname_img)

    # close the plot handle
    plt.close('all')


''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''
