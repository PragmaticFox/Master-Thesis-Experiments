#!/bin/python3

import math
import random
import pathlib

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# local import
import helper

identifier_string = "IK_2d_"

string_title_joint_angles_plot = f'\nJoint Angles in Degrees\n2D Two-Linkage Robot Inverse Kinematics\n'

string_title_heatmap_plot = f'\nTerminal Energy Landscape in Meters\n2D Two-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_plot = f'\nJacobian Frobenius Norm Landscape\n2D Two-Linkage Robot Inverse Kinematics\n'

string_title_heatmap_histogram = f'\nTerminal Energy Histogram\n2D Two-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_histogram = f'\nJacobian Frobenius Norm Histogram\n2D Two-Linkage Robot Inverse Kinematics\n'

N_DIM_THETA = 2
N_DIM_X = 2

N_TRAJOPT = 1

N_DIM_X_STATE = 1*N_DIM_X


LR_INITIAL = 1e-2

# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99925 # for 10k
# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99975 # for 30k
LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99985  # for 50k
# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.999925 # for 100k

FK_ORIGIN = [0.0, 0.0]

RADIUS_INNER = 0.0
RADIUS_OUTER = 1.0

SAMPLE_CIRCLE = True

LIMITS = [[-1.0, 1.0], [-1.0, 1.0]]

LIMITS_PLOTS = LIMITS
LIMITS_PLOTS = [[-1.0, 1.0], [-1.0, 1.0]]

LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]

CONSTRAINTS = [[0.0, 2.0*math.pi], [0.0, 2.0*math.pi]]

''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


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


def compute_and_save_joint_angles_plot(model, device, dpi, n_one_dim, dir_path_img, index, fname_img, title_string):

    alpha = 0.5

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

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=1, hspace=0.25)

    c = 0

    for i in range(2):

        axes[i].set_aspect(aspect='equal', adjustable='box')
        axes[i].set_title(
            f'\nJoint {i+1}\n',
            fontdict={'fontsize': 8, 'fontweight': 'normal',
                      'horizontalalignment': 'center'},
            pad=5
        )
        axes[i].axis([dimX.min(), dimX.max(), dimY.min(), dimY.max()])
        c = axes[i].pcolormesh(dimX, dimY, theta_hat[:, :,  -1, i],
                               cmap='RdYlBu', shading='gouraud', vmin=rad_min, vmax=rad_max)
        #c = axes[i].pcolormesh(dimX, dimY, theta_hat[:, :,  -1, i], cmap = 'twilight', shading = 'gouraud', vmin = rad_min, vmax = rad_max)

    cb = fig.colorbar(c, ax=axes.ravel().tolist(), extend='max')

    plt.suptitle(
        title_string,
        fontsize=10,
        fontweight='normal',
        horizontalalignment='center',
        x=0.825,
        y=1.515
    )

    if SAMPLE_CIRCLE:
        circleInner1 = plt.Circle(
            (0.0, 0.0), radius=RADIUS_INNER, color='orange', fill=False, lw=4.0, alpha=alpha)
        circleOuter1 = plt.Circle(
            (0.0, 0.0), radius=RADIUS_OUTER, color='orange', fill=False, lw=4.0, alpha=alpha)
        circleInner2 = plt.Circle(
            (0.0, 0.0), radius=RADIUS_INNER, color='orange', fill=False, lw=4.0, alpha=alpha)
        circleOuter2 = plt.Circle(
            (0.0, 0.0), radius=RADIUS_OUTER, color='orange', fill=False, lw=4.0, alpha=alpha)

    if LIMITS_PLOTS != LIMITS:
        rectangle1 = plt.Rectangle(xy=(LIMITS[0][0], LIMITS[1][0]), width=LIMITS[0][1]-LIMITS[0]
                                   [0], height=LIMITS[1][1]-LIMITS[1][0], color='orange', fill=False, lw=4.0, alpha=alpha)
        rectangle2 = plt.Rectangle(xy=(LIMITS[0][0], LIMITS[1][0]), width=LIMITS[0][1]-LIMITS[0]
                                   [0], height=LIMITS[1][1]-LIMITS[1][0], color='orange', fill=False, lw=4.0, alpha=alpha)

    if LIMITS_PLOTS != LIMITS or SAMPLE_CIRCLE:

        if SAMPLE_CIRCLE:
            axes[0].add_patch(circleInner1)
            axes[0].add_patch(circleOuter1)
            axes[1].add_patch(circleInner2)
            axes[1].add_patch(circleOuter2)

        if LIMITS_PLOTS != LIMITS:
            axes[0].add_patch(rectangle1)
            axes[1].add_patch(rectangle2)

    helper.save_figure(fig, dpi, dir_path_img, str(index) + "_" + fname_img)
    helper.save_figure(fig, dpi, "", fname_img)

    # close the plot handle
    plt.close()


def compute_and_save_jacobian_plot(model, device, X_state_train, dpi, n_one_dim, dir_path_img, index, fname_img, fontdict, title_string):

    X_state_train = X_state_train.detach().cpu()

    alpha = 0.5
    alpha_train_samples = 0.25

    dimX = np.linspace(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1], n_one_dim)
    dimY = np.linspace(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1], n_one_dim)

    dimX, dimY = np.meshgrid(dimX, dimY)

    x_state = torch.tensor(np.stack(
        (dimX.flatten(), dimY.flatten()), axis=-1), requires_grad=True).to(device)

    jac = torch.zeros(
        size=(n_one_dim*n_one_dim, N_TRAJOPT*N_DIM_THETA, N_DIM_X))

    for i in range(n_one_dim*n_one_dim):
        jac[i] = torch.reshape(torch.autograd.functional.jacobian(
            model, x_state[i:i+1], create_graph=False, strict=False), shape=(N_TRAJOPT*N_DIM_THETA, N_DIM_X))

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
                      norm=matplotlib.colors.LogNorm(vmin=jac_norm_min, vmax=jac_norm_max))

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], ms=2.0,
            marker='o', color='k', ls='', alpha=alpha_train_samples)

    cb = fig.colorbar(c, ax=ax, extend='max')

    if SAMPLE_CIRCLE:
        circleInner = plt.Circle(
            (0.0, 0.0), radius=RADIUS_INNER, color='orange', fill=False, lw=4.0, alpha=alpha)
        circleOuter = plt.Circle(
            (0.0, 0.0), radius=RADIUS_OUTER, color='orange', fill=False, lw=4.0, alpha=alpha)

    if LIMITS_PLOTS != LIMITS:
        rectangle = plt.Rectangle(xy=(LIMITS[0][0], LIMITS[1][0]), width=LIMITS[0][1]-LIMITS[0]
                                  [0], height=LIMITS[1][1]-LIMITS[1][0], color='orange', fill=False, lw=4.0, alpha=alpha)

    legend_entries = []

    if LIMITS_PLOTS != LIMITS or SAMPLE_CIRCLE:

        legend_entries = legend_entries + \
            [matplotlib.patches.Patch(
                color='orange', alpha=alpha, label='Sampling Area')]

        if SAMPLE_CIRCLE:
            ax.add_patch(circleInner)
            ax.add_patch(circleOuter)

        if LIMITS_PLOTS != LIMITS:
            ax.add_patch(rectangle)

    plt.legend(loc='upper right', handles=legend_entries)

    helper.save_figure(fig, dpi, dir_path_img, str(index) + "_" + fname_img)
    helper.save_figure(fig, dpi, "", fname_img)

    # close the plot handle
    plt.close('all')


def compute_and_save_heatmap_plot(model, device, X_state_train, metrics, dpi, is_constrained, n_one_dim, dir_path_img, index, fname_img, fontdict, title_string):

    X_state_train = X_state_train.detach().cpu()

    test_terminal_energy_mean = metrics[0].detach().cpu()

    alpha = 0.5
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
                      norm=matplotlib.colors.LogNorm(vmin=terminal_energy_min, vmax=terminal_energy_max))

    ax.plot(X_state_train[:, 0], X_state_train[:, 1], ms=2.0,
            marker='o', color='k', ls='', alpha=alpha_train_samples)

    cb = fig.colorbar(c, ax=ax, extend='max')
    cb.ax.plot([0, 1], [test_terminal_energy_mean]*2, 'k', alpha=alpha, lw=8.0)

    if SAMPLE_CIRCLE:
        circleInner = plt.Circle(
            (0.0, 0.0), radius=RADIUS_INNER, color='orange', fill=False, lw=4.0, alpha=alpha)
        circleOuter = plt.Circle(
            (0.0, 0.0), radius=RADIUS_OUTER, color='orange', fill=False, lw=4.0, alpha=alpha)

    if LIMITS_PLOTS != LIMITS:
        rectangle = plt.Rectangle(xy=(LIMITS[0][0], LIMITS[1][0]), width=LIMITS[0][1]-LIMITS[0]
                                  [0], height=LIMITS[1][1]-LIMITS[1][0], color='orange', fill=False, lw=4.0, alpha=alpha)

    legend_entries = [
        matplotlib.lines.Line2D([0], [0], lw=0.0, marker='o', color='k',
                                alpha=alpha_train_samples, markersize=10.0, label='Train Samples'),
        matplotlib.patches.Patch(
            color='k', alpha=alpha, label='Test Mean ± Std')
    ]

    if LIMITS_PLOTS != LIMITS or SAMPLE_CIRCLE:

        legend_entries = legend_entries + \
            [matplotlib.patches.Patch(
                color='orange', alpha=alpha, label='Sampling Area')]

        if SAMPLE_CIRCLE:
            ax.add_patch(circleInner)
            ax.add_patch(circleOuter)

        if LIMITS_PLOTS != LIMITS:
            ax.add_patch(rectangle)

    plt.legend(loc='upper right', handles=legend_entries)

    helper.save_figure(fig, dpi, dir_path_img, str(index) + "_" + fname_img)
    helper.save_figure(fig, dpi, "", fname_img)

    # close the plot handle
    plt.close('all')


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

