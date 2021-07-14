#!/bin/python3

import os
import math
import torch
import shutil
import pathlib
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# fixes a possible "Fail to allocate bitmap" issue
# https://github.com/matplotlib/mplfinance/issues/386#issuecomment-869950969
matplotlib.use("Agg")

# local import
import helper

identifier_string = "IK_2d_"

string_title_joint_angles_plot = f'\nJoint Angles in Degrees\n2D Two-Linkage Robot Inverse Kinematics\n'

string_title_terminal_energy_plot = f'\nTerminal Energy Landscape in Meters\n2D Two-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_plot = f'\nJacobian Frobenius Norm Landscape\n2D Two-Linkage Robot Inverse Kinematics\n'

string_title_terminal_energy_histogram = f'\nTerminal Energy Histogram\n2D Two-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_histogram = f'\nJacobian Frobenius Norm Histogram\n2D Two-Linkage Robot Inverse Kinematics\n'

N_DIM_THETA = 2
N_DIM_JOINTS = N_DIM_THETA

N_DIM_X = 2

N_TRAJOPT = 1

N_DIM_X_STATE = 1*N_DIM_X

FK_ORIGIN = [0.0, 0.0]

RADIUS_INNER = 0.0
RADIUS_OUTER = 1.0

SAMPLE_CIRCLE = True

LIMITS = [[-1.0, 1.0], [-1.0, 1.0]]
LIMITS_PLOTS = LIMITS

LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]

CONSTRAINTS = [[0.0, 2.0*math.pi]] * N_DIM_THETA

''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


def save_script(directory):

    # saves a copy of the current python script into the folder
    shutil.copy(__file__, pathlib.Path(directory, os.path.basename(__file__)))


def fk(theta):

    theta_accum = torch.zeros((theta.shape[0])).to(theta.device)
    p = torch.zeros(
        (theta.shape[0], theta.shape[1] + 1, N_DIM_X)).to(theta.device)

    for i in range(theta.shape[1]):

        theta_accum = theta_accum + theta[:, i]

        p[:, i+1, 0] = p[:, i, 0] + float(LENGTHS[i]) * torch.cos(theta_accum)
        p[:, i+1, 1] = p[:, i, 1] + float(LENGTHS[i]) * torch.sin(theta_accum)

    return p[:, 1:, :]


def visualize_trajectory_and_save_image(x_state, x_hat_fk_chain, dir_path_img, fname_img):

    plt.scatter(x_state[0], x_state[1], c='r', s=100, zorder=-10)

    for t in range(N_TRAJOPT):

        plt.arrow(
            FK_ORIGIN[0],
            FK_ORIGIN[1],
            x_hat_fk_chain[0, t, 0] - FK_ORIGIN[0],
            x_hat_fk_chain[0, t, 1] - FK_ORIGIN[1],
            length_includes_head=True,
            width=0.005,
            head_length=0.015,
            head_width=0.025,
            fc='tab:blue',
            ec='tab:blue'
        )

        for i in range(1, len(x_hat_fk_chain), 1):

            plt.scatter(
                x_hat_fk_chain[i-1, t, 0], x_hat_fk_chain[i-1, t, 1], c='0.5', s=5)

            plt.arrow(
                x_hat_fk_chain[i-1, t, 0],
                x_hat_fk_chain[i-1, t, 1],
                x_hat_fk_chain[i, t, 0] - x_hat_fk_chain[i-1, t, 0],
                x_hat_fk_chain[i, t, 1] - x_hat_fk_chain[i-1, t, 1],
                length_includes_head=True,
                width=0.0075,
                head_length=0.015,
                head_width=0.025,
                fc='tab:blue',
                ec='tab:blue'
            )

        plt.scatter(x_hat_fk_chain[-1, t, 0],
                    x_hat_fk_chain[-1, t, 1], c='k', s=25)

    plt.scatter(FK_ORIGIN[0], FK_ORIGIN[1], c='0.5', s=5)

    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.xlabel("x")
    plt.ylabel("y")

    plt.gca().set_aspect('equal', adjustable='box')

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close()


def compute_and_save_samples_plot(X_state_train, X_state_val, X_state_test, dir_path_img, fname_img):

    plt.plot(X_state_train[:, 0], X_state_train[:, 1],
             ms=1.0, marker='o', color='b', ls='')
    plt.plot(X_state_val[:, 0], X_state_val[:, 1],
             ms=1.0, marker='o', color='g', ls='')
    plt.plot(X_state_test[:, 0], X_state_test[:, 1],
             ms=1.0, marker='o', color='r', ls='')

    plt.xlabel("x")
    plt.ylabel("y")

    plt.gca().set_aspect('equal', adjustable='box')

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    plt.close()


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
        #constraint_bound = soft_bound_constraint(lower_limit = 0.0, upper_limit = math.pi, eps_rel = 1e-1, stiffness = 1e-0, x = theta_hat[:, -1])

    energy += constraint_bound

    return energy, constraint_bound, terminal_position_distance, x_hat_fk_chain


def compute_and_save_joint_angles_plot(rng, model, device, X_state_train, dpi, n_one_dim, dir_path_img, fname_img, fontdict, title_string):

    xs__ = X_state_train[:, 0].detach().cpu()
    ys__ = X_state_train[:, 1].detach().cpu()

    alpha_train_samples = 0.25

    dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
    dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(
        np.stack((dimX.flatten(), dimY.flatten()), axis=-1)).to(device)

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
            f'\nJoint {j+1}\n' + title_string,
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

    X_state_train = X_state_train.detach().cpu()

    alpha_train_samples = 0.25

    dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
    dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(np.stack(
        (dimX.flatten(), dimY.flatten()), axis=-1), requires_grad=True).to(device)

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
        title_string,
        fontdict=fontdict,
        pad=5
    )

    ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
    c = ax.pcolormesh(dimX, dimY, jac_norm, cmap='RdBu', shading='gouraud',
                      norm=matplotlib.colors.LogNorm(vmin=helper.COLORBAR_JACOBIAN_LOWER_THRESHOLD, vmax=helper.COLORBAR_JACOBIAN_UPPER_THRESHOLD))

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], ms=2.0,
            marker='o', color='k', ls='', alpha=alpha_train_samples)

    cb = fig.colorbar(c, ax=ax, extend='max')

    plt.xlabel("x")
    plt.ylabel("y")

    helper.save_figure(fig, dpi, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_terminal_energy_plot(rng, model, device, X_state_train, dpi, is_constrained, n_one_dim, dir_path_img, fname_img, fontdict, title_string):

    X_state_train = X_state_train.detach().cpu()

    alpha_train_samples = 0.25

    dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
    dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(
        np.stack((dimX.flatten(), dimY.flatten()), axis=-1)).to(device)

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
        title_string,
        fontdict=fontdict,
        pad=5
    )

    ax.axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
    c = ax.pcolormesh(dimX, dimY, terminal_energy, cmap='RdBu', shading='gouraud',
                      norm=matplotlib.colors.LogNorm(vmin=helper.COLORBAR_ENERGY_LOWER_THRESHOLD, vmax=helper.COLORBAR_ENERGY_UPPER_THRESHOLD))

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], ms=2.0,
            marker='o', color='k', ls='', alpha=alpha_train_samples)

    cb = fig.colorbar(c, ax=ax, extend='max')

    plt.xlabel("x")
    plt.ylabel("y")

    helper.save_figure(fig, dpi, dir_path_img, fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_jacobian_histogram(rng, model, X_samples, dpi, dir_path_img, fname_img, fontdict, title_string):

    n_samples = X_samples.shape[0]

    model_sum = lambda x : torch.sum(model(x), axis = 0)

    jac = torch.zeros(size=(n_samples, N_TRAJOPT * N_DIM_THETA, N_DIM_X)).to(X_samples.device)
    jac = torch.autograd.functional.jacobian(model_sum, X_samples, create_graph = False, strict = False, vectorize = True).permute(1, 0, 2)

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


def compute_and_save_joint_angles_region_plot(rng, device, n_samples_theta, dpi, dir_path_img, fname_img):

    theta = torch.tensor([helper.sample_joint_angles(rng, CONSTRAINTS) for _ in range(
        n_samples_theta)], dtype=helper.DTYPE_TORCH).to(device)

    x_fk_chain = fk(theta)

    x_fk_chain = torch.reshape(input=x_fk_chain, shape=(
        n_samples_theta, N_TRAJOPT, N_DIM_JOINTS, N_DIM_X_STATE))
    x_fk_chain = torch.transpose(input=x_fk_chain, dim0=1, dim1=2)
    x_fk_chain = x_fk_chain.detach().cpu()

    xs = x_fk_chain[:, -1, -1, 0]
    ys = x_fk_chain[:, -1, -1, 1]

    xs_min = xs.min()
    xs_max = xs.max()

    ys_min = ys.min()
    ys_max = ys.max()

    x_min = min(xs_min, LIMITS[0][0])
    x_max = max(xs_max, LIMITS[0][1])

    y_min = min(ys_min, LIMITS[1][0])
    y_max = max(ys_max, LIMITS[1][1])

    fig, ax = plt.subplots()

    ax.plot(xs, ys, ms=1.0, marker='o', color='b', ls='', alpha=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("x")
    ax.set_ylabel("y")


    ax.set_title(
        f"\nx = [{xs_min}, {xs_max}]\ny = [{ys_min}, {ys_max}]\n",
        # fontdict=fontdict,
        pad=5
    )

    ax.set_aspect('auto', adjustable='box')

    helper.save_figure(fig, dpi, dir_path_img,
                       identifier_string + "joint_angles_region_plot.png")

    # close the plot handle
    plt.close('all')

