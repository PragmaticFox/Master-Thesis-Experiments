#!/bin/python3

import math
import random
import pathlib

import torch
import numpy as np

import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# local import
import helper

IS_UR5_ROBOT = True

identifier_string = "IK_3d_"

if IS_UR5_ROBOT :

    identifier_string += "UR5_"

string_title_joint_angles_plot = f'\nJoint Angles in Degrees\n3D Three-Linkage Robot Inverse Kinematics\n'

string_title_heatmap_plot = f'\nTerminal Energy Landscape in Meters\n3D Three-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_plot = f'\nJacobian Frobenius Norm Landscape\n3D Three-Linkage Robot Inverse Kinematics\n'

string_title_heatmap_histogram = f'\nTerminal Energy Histogram\n3D Three-Linkage Robot Inverse Kinematics\n'
string_title_jacobian_histogram = f'\nJacobian Frobenius Norm Histogram\n3D Three-Linkage Robot Inverse Kinematics\n'

N_DIM_THETA = 3

if IS_UR5_ROBOT :

    N_DIM_THETA = 12

N_DIM_JOINTS = N_DIM_THETA

if IS_UR5_ROBOT :

    N_DIM_JOINTS = N_DIM_THETA // 2

N_DIM_X = 3

N_TRAJOPT = 1

N_DIM_X_STATE = 1*N_DIM_X

LR_INITIAL = 1e-2

# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99925 # for 10k
LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99950 # for 25k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99975 # for 30k
#LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.99985  # for 50k
# LR_SCHEDULER_MULTIPLICATIVE_REDUCTION = 0.999925 # for 100k

FK_ORIGIN = [0.0, 0.0, 0.0]

RADIUS_INNER = 0.0
RADIUS_OUTER = 1.0

SAMPLE_CIRCLE = True

LIMITS = [[-0.5, 0.5], [-0.1, 0.1], [0.5, 0.75]]

if IS_UR5_ROBOT :

    LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

LIMITS_PLOTS = LIMITS
#LIMITS_PLOTS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

if IS_UR5_ROBOT :

    LIMITS_PLOTS = LIMITS
    #LIMITS_PLOTS = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]


LENGTHS = N_DIM_THETA*[1.0/N_DIM_THETA]

CONSTRAINTS = [[0.0, 2.0*math.pi]] * N_DIM_THETA

if IS_UR5_ROBOT :

    CONSTRAINTS = [[0.0, 2.0*math.pi]] * N_DIM_THETA

''' ---------------------------------------------- CLASSES & FUNCTIONS ---------------------------------------------- '''


def dh_matrix(n, theta, alpha, d, r) :

    device = theta.device

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
    n_batch_times_n_trajOpt = theta.shape[0]
    n_dim_theta = theta.shape[1]

    if IS_UR5_ROBOT :

        transform = torch.reshape(torch.eye(4,4), shape = (1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta//2 + 1, 1, 1).to(device)

        transform[:, 1] = torch.matmul(transform[:, 0].clone(), dh_matrix(n_batch_times_n_trajOpt, theta[:, 0], theta[:, 1], 0.089159, 0.0).clone())
        transform[:, 2] = torch.matmul(transform[:, 1].clone(), dh_matrix(n_batch_times_n_trajOpt, theta[:, 2], theta[:, 3], 0.0, -0.425).clone())
        transform[:, 3] = torch.matmul(transform[:, 2].clone(), dh_matrix(n_batch_times_n_trajOpt, theta[:, 4], theta[:, 5], 0.0, -0.39225).clone())
        transform[:, 4] = torch.matmul(transform[:, 3].clone(), dh_matrix(n_batch_times_n_trajOpt, theta[:, 6], theta[:, 7], 0.10915, 0.0).clone())
        transform[:, 5] = torch.matmul(transform[:, 4].clone(), dh_matrix(n_batch_times_n_trajOpt, theta[:, 8], theta[:, 9], 0.09465, 0.0).clone())
        transform[:, 6] = torch.matmul(transform[:, 5].clone(), dh_matrix(n_batch_times_n_trajOpt, theta[:, 10], theta[:, 11], 0.0823, 0.0).clone())

        p  = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
        p_final = torch.reshape(torch.tensor([0.0, 0.0, 0.0, 1.0]), shape = (1, 1, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta//2, 1).to(device)

        for i in range(n_dim_theta//2) :
                
            p_final[:, i] = torch.matmul(torch.clone(transform[:, i+1]), p)

        #return p_final[:, :, :-1], transform[:, 6, :3, :3]
        return p_final[:, :, :-1]

    p = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)
    p_final = torch.reshape(torch.tensor([0.0, 0.0, 0.0, 1.0]), shape=(1, 1, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1).to(device)
    rt_hom = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)
    r_hom_i = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)
    t_hom_i = torch.reshape(torch.eye(4, 4), shape=(1, 1, 4, 4)).repeat(n_batch_times_n_trajOpt, n_dim_theta+1, 1, 1).to(device)

    for i in range(N_DIM_THETA):
        '''
        if (i % 3 == 0):

            # rotation around x-axis (yz-plane)
            # homogeneous coordinates

            #t_hom_i[:, i, 0, 3] = LENGTHS[i]
            t_hom_i[:, i, 1, 3] = LENGTHS[i]
            #t_hom_i[:, i, 2, 3] = LENGTHS[i]

            r_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])
            r_hom_i[:, i, 1, 2] = -torch.sin(theta[:, i])
            r_hom_i[:, i, 2, 1] = torch.sin(theta[:, i])
            r_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])
        '''

        if (i % 3 < 2):

            # rotation around y-axis (xz-plane)
            # homogeneous coordinates

            t_hom_i[:, i, 0, 3] = LENGTHS[i]
            #t_hom_i[:, i, 1, 3] = LENGTHS[i]
            #t_hom_i[:, i, 2, 3] = LENGTHS[i]

            r_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            r_hom_i[:, i, 0, 2] = torch.sin(theta[:, i])
            r_hom_i[:, i, 2, 0] = -torch.sin(theta[:, i])
            r_hom_i[:, i, 2, 2] = torch.cos(theta[:, i])

        if (i % 3 == 2):

            # rotation around z-axis (xy-plane)
            # homogeneous coordinates

            t_hom_i[:, i, 0, 3] = LENGTHS[i]
            #t_hom_i[:, i, 1, 3] = LENGTHS[i]
            #t_hom_i[:, i, 2, 3] = LENGTHS[i]

            r_hom_i[:, i, 0, 0] = torch.cos(theta[:, i])
            r_hom_i[:, i, 0, 1] = -torch.sin(theta[:, i])
            r_hom_i[:, i, 1, 0] = torch.sin(theta[:, i])
            r_hom_i[:, i, 1, 1] = torch.cos(theta[:, i])

        #print(t_hom_i[0])
        #print(r_hom_i[0])
        
        tmp = torch.matmul(torch.clone(rt_hom[:, i]), torch.clone(r_hom_i[:, i]))
        rt_hom[:, i+1] = torch.matmul(torch.clone(tmp), torch.clone(t_hom_i[:, i]))
        
        #tmp = torch.matmul(torch.clone(r_hom_i[:, i]), torch.clone(t_hom_i[:, i]))
        #rt_hom[:, i+1] = torch.matmul(torch.clone(tmp), torch.clone(rt_hom[:, i]))       
        p_final[:, i+1] = torch.matmul(rt_hom[:, i+1], p)

    return p_final[:, 1:, :-1]


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
            normalize = False
        )

        for i in range(1, N_DIM_JOINTS, 1):

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
                normalize = False
            )

        ax.scatter(x_hat_fk_chain[-1, t, 0], x_hat_fk_chain[-1, t, 1], x_hat_fk_chain[-1, t, 2], c='k', s=10)

    ax.scatter(FK_ORIGIN[0], FK_ORIGIN[1], FK_ORIGIN[2], c='0.5', s=10)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.gca().set_aspect('auto', adjustable='box')

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, "", fname_img)

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

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, dir_path_img, fname_img)

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


def compute_and_save_joint_angles_plot(model, device, dpi, n_one_dim, dir_path_img, index, fname_img, title_string):

    return


def compute_and_save_jacobian_plot(model, device, X_state_train, dpi, n_one_dim, dir_path_img, index, fname_img, fontdict, title_string):

    return


def compute_and_save_heatmap_plot(rng, model, device, X_state_train, metrics, dpi, is_constrained, n_one_dim, dir_path_img, index, fname_img, fontdict, title_string):

    X_state_train = X_state_train.detach().cpu()

    X_state_train = X_state_train[torch.sqrt(X_state_train[:, 2] ** 2) < 1e-3]

    test_terminal_energy_mean = metrics[0].detach().cpu()

    n_samples = 25000

    alpha = 0.5
    alpha_train_samples = 0.25

    x_state = torch.tensor([helper.compute_sample(rng, LIMITS, SAMPLE_CIRCLE, RADIUS_OUTER, RADIUS_INNER) for _ in range(n_samples)], dtype=helper.DTYPE_TORCH).to(device)

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
    terminal_energy_min = terminal_energy.min()
    terminal_energy_max = terminal_energy.max()

    dimX = x_state[:, 0].detach().cpu()
    dimY = x_state[:, 1].detach().cpu()
    dimZ = x_state[:, 2].detach().cpu()

    # plot

    #fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')

    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=1, hspace=1)

    ax.set_aspect(aspect='auto', adjustable='box')

    ax.set_title(
        title_string,
        fontdict=fontdict,
        pad=5
    )

    cmap = pl.cm.RdBu
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.flip(np.logspace(-1.5, 0, cmap.N))
    #print(my_cmap[:,-1])
    my_cmap = ListedColormap(my_cmap)

    c = ax.scatter(
        xs = dimX,
        ys = dimY,
        zs = dimZ,
        zdir = 'z',
        s = 20,
        c = terminal_energy,
        depthshade = True,
        cmap = my_cmap,
        norm = matplotlib.colors.LogNorm(vmin = terminal_energy_min, vmax = terminal_energy_max)
        #alpha = 0.75
    )

    ax.set_xlim(LIMITS_PLOTS[0][0], LIMITS_PLOTS[0][1])
    ax.set_ylim(LIMITS_PLOTS[1][0], LIMITS_PLOTS[1][1])
    ax.set_zlim(LIMITS_PLOTS[2][0], LIMITS_PLOTS[2][1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig = plt.gcf()

    cb = fig.colorbar(c, ax=ax, extend='max')
    cb.ax.plot([0, 1], [test_terminal_energy_mean]*2, 'k', alpha=alpha, lw=8.0)

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


def compute_and_save_joint_angles_region_plot(device, n_samples_theta, dpi, dir_path_img, fname_img):

    theta = torch.tensor([helper.sample_joint_angles(random, CONSTRAINTS) for _ in range(n_samples_theta)], dtype = helper.DTYPE_TORCH).to(device)

    x_fk_chain = fk(theta)

    x_fk_chain = torch.reshape(input = x_fk_chain, shape = (n_samples_theta, N_TRAJOPT, N_DIM_JOINTS, N_DIM_X_STATE))
    x_fk_chain = torch.transpose(input = x_fk_chain, dim0 = 1, dim1 = 2)
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

    ax = plt.axes(projection = '3d')

    ax.plot(xs, ys, zs, ms = 1.0, marker = 'o', color = 'b', ls = '', alpha = 0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(
        f"\nx = [{xs_min}, {xs_max}]\ny = [{ys_min}, {ys_max}]\nz = [{zs_min}, {zs_max}]\n",
        #fontdict=fontdict,
        pad=5
    )

    plt.gca().set_aspect('auto', adjustable='box')

    helper.save_figure(plt.gcf(), helper.SAVEFIG_DPI, "", identifier_string + "joint_angles_region_plot_3d.png")

    plt.close('all')

    n_slices = 10.0

    delta = (zs_max - zs_min) / n_slices

    for i in range(int(n_slices)) :

        ax = plt.axes()

        indices_max = zs <= zs_min + (i+1)*delta

        xs_ = xs[indices_max]
        ys_ = ys[indices_max]
        zs_ = zs[indices_max]

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

        ax.plot(xs__, ys__, ms = 1.0, marker = 'o', color = 'b', ls = '', alpha = 0.5)

        x_min__ = min(xs_min__, LIMITS[0][0])
        x_max__ = max(xs_max__, LIMITS[0][1])

        y_min__ = min(ys_min__, LIMITS[1][0])
        y_max__ = max(ys_max__, LIMITS[1][1])

        z_min__ = min(zs_min__, LIMITS[2][0])
        z_max__ = max(zs_max__, LIMITS[2][1])

        ax.set_xlim(x_min__, x_max__)
        ax.set_ylim(y_min__, y_max__)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
        ax.set_title(
            f"\nx = [{xs_min__}, {xs_max__}]\ny = [{ys_min__}, {ys_max__}]\nz = [{zs_min__}, {zs_max__}]\n",
            #fontdict=fontdict,
            pad=5
        )

        plt.gca().set_aspect('auto', adjustable='box')

        fig = plt.gcf()

        helper.save_figure(fig, dpi, "", str(i+1) + "_" + fname_img)
        helper.save_figure(fig, dpi, dir_path_img, str(i+1) + "_" + fname_img)

        plt.close('all')

