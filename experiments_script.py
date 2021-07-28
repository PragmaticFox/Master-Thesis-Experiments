#!/bin/python3

import pathlib

benchmark_path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), "benchmark.py")
compute_hist_grom_model_path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), "compute_hist_from_model.py")


#'''
# Experiment: The RNG Matters

local_globals_list = [
    {
        "exp_SAMPLING_MODE": 0
    },
    {
        "exp_SAMPLING_MODE": 1
    },
    {
        "exp_SAMPLING_MODE": 2
    }
]

for start_seed in range(25) :
    for local_globals in local_globals_list :
        with open(benchmark_path) as f:
            code = compile(f.read(), benchmark_path, "exec")
            globals().update(local_globals)
            globals()["__file__"] = benchmark_path
            globals()["exp_START_SEED"] = start_seed
            globals()["exp_N_SAMPLES_TRAIN"] = 1000
            globals()["exp_N_ITERATIONS"] = 100
            globals()["exp_IS_MODE_2_ABLATION"] = False
            globals()["exp_IS_TWOLINKAGE_CONSTRAINED"] = False
            exec(code, globals())

#'''

'''

# Experiment: Two-Linkage Main Experiments

local_globals_list = [
    {
        "exp_SAMPLING_MODE": 1,
        "exp_N_SAMPLES_TRAIN": 1000
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_N_SAMPLES_TRAIN": 1000
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_N_SAMPLES_TRAIN": 10000
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_N_SAMPLES_TRAIN": 10000
    },
    {
        "exp_SAMPLING_MODE": 0,
        "exp_N_SAMPLES_TRAIN": 1000
    },
    {
        "exp_SAMPLING_MODE": 0,
        "exp_N_SAMPLES_TRAIN": 10000
    },
    {
        "exp_SAMPLING_MODE": 0,
        "exp_N_SAMPLES_TRAIN": 100
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_N_SAMPLES_TRAIN": 100
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_N_SAMPLES_TRAIN": 100
    },
    {
        "exp_SAMPLING_MODE": 0,
        "exp_N_SAMPLES_TRAIN": 10
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_N_SAMPLES_TRAIN": 10
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_N_SAMPLES_TRAIN": 10
    }
]

for local_globals in local_globals_list :
    with open(benchmark_path) as f:
        code = compile(f.read(), benchmark_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = benchmark_path
        globals()["exp_N_ITERATIONS"] = 20000
        globals()["exp_IS_MODE_2_ABLATION"] = False
        globals()["exp_IS_TWOLINKAGE_CONSTRAINED"] = False
        exec(code, globals())

'''



'''
# Experiment: fix histograms for two-linkage experiments

local_globals_list = [
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10_Mode_0_Iterations_10k_2021_07_22_15_19_45/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10_Mode_0_Iterations_10k_2021_07_22_15_19_45/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10_Mode_1_Iterations_10k_2021_07_22_16_04_12/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10_Mode_1_Iterations_10k_2021_07_22_16_04_12/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10_Mode_2_Iterations_10k_2021_07_22_17_39_30/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10_Mode_2_Iterations_10k_2021_07_22_17_39_30/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_100_Mode_0_Iterations_10k_2021_07_22_15_21_57/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_100_Mode_0_Iterations_10k_2021_07_22_15_21_57/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_100_Mode_1_Iterations_10k_2021_07_22_16_06_31/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_100_Mode_1_Iterations_10k_2021_07_22_16_06_31/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_100_Mode_2_Iterations_10k_2021_07_22_17_44_52/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_100_Mode_2_Iterations_10k_2021_07_22_17_44_52/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_1000_Mode_0_Iterations_10k_2021_07_22_15_29_52/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_1000_Mode_0_Iterations_10k_2021_07_22_15_29_52/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_1000_Mode_1_Iterations_10k_2021_07_22_16_09_08/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_1000_Mode_1_Iterations_10k_2021_07_22_16_09_08/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_1000_Mode_2_Iterations_10k_2021_07_22_17_47_07/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_1000_Mode_2_Iterations_10k_2021_07_22_17_47_07/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10000_Mode_0_Iterations_10k_2021_07_22_15_38_52/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10000_Mode_0_Iterations_10k_2021_07_22_15_38_52/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10000_Mode_1_Iterations_10k_2021_07_22_16_14_47/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10000_Mode_1_Iterations_10k_2021_07_22_16_14_47/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10000_Mode_2_Iterations_10k_2021_07_22_17_49_44/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/two_linkage_sampling_comparisons/IK_2d_Samples_10000_Mode_2_Iterations_10k_2021_07_22_17_49_44/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/100k_iterations_comparison/IK_2d_Samples_1000_Mode_0_Iterations_100k_2021_07_22_15_47_34/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/100k_iterations_comparison/IK_2d_Samples_1000_Mode_0_Iterations_100k_2021_07_22_15_47_34/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/100k_iterations_comparison/IK_2d_Samples_1000_Mode_1_Iterations_100k_2021_07_22_16_51_36/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/100k_iterations_comparison/IK_2d_Samples_1000_Mode_1_Iterations_100k_2021_07_22_16_51_36/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/100k_iterations_comparison/IK_2d_Samples_1000_Mode_2_Iterations_100k_2021_07_22_17_58_37/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/100k_iterations_comparison/IK_2d_Samples_1000_Mode_2_Iterations_100k_2021_07_22_17_58_37/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/expansion_sampling_ablation/IK_2d_Samples_1000_Mode_2_Iterations_10k_2021_07_22_17_47_07/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/expansion_sampling_ablation/IK_2d_Samples_1000_Mode_2_Iterations_10k_2021_07_22_17_47_07/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/expansion_sampling_ablation/IK_2d_Samples_1000_Mode_2a_Iterations_10k_2021_07_22_18_19_01/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/expansion_sampling_ablation/IK_2d_Samples_1000_Mode_2a_Iterations_10k_2021_07_22_18_19_01/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/fixing_artifacts_with_constraints/IK_2d_Samples_1000_Mode_0c_Iterations_10k_2021_07_22_18_13_56/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/fixing_artifacts_with_constraints/IK_2d_Samples_1000_Mode_0c_Iterations_10k_2021_07_22_18_13_56/plots",
        "exp_IS_TWOLINKAGE_CONSTRAINED": True
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/fixing_artifacts_with_constraints/IK_2d_Samples_1000_Mode_0_Iterations_10k_2021_07_22_15_29_52/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/fixing_artifacts_with_constraints/IK_2d_Samples_1000_Mode_0_Iterations_10k_2021_07_22_15_29_52/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/jacobian_frobenius_norm_as_an_alternative/IK_2d_Samples_1000_Mode_0_Iterations_100k_2021_07_22_15_47_34/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/jacobian_frobenius_norm_as_an_alternative/IK_2d_Samples_1000_Mode_0_Iterations_100k_2021_07_22_15_47_34/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/jacobian_frobenius_norm_as_an_alternative/IK_2d_Samples_1000_Mode_0c_Iterations_100k_2021_07_22_18_22_14/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/jacobian_frobenius_norm_as_an_alternative/IK_2d_Samples_1000_Mode_0c_Iterations_100k_2021_07_22_18_22_14/plots",
        "exp_IS_TWOLINKAGE_CONSTRAINED": True
    },
]

for local_globals in local_globals_list :
    with open(compute_hist_grom_model_path) as f:
        code = compile(f.read(), compute_hist_grom_model_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = compute_hist_grom_model_path
        exec(code, globals())


'''

'''

# Experiment: fix histograms for three-linkage experiments

local_globals_list = [
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10_Mode_0_Iterations_10k_2021_07_21_23_44_54/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10_Mode_0_Iterations_10k_2021_07_21_23_44_54/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10_Mode_1_Iterations_10k_2021_07_22_00_27_11/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10_Mode_1_Iterations_10k_2021_07_22_00_27_11/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10_Mode_2_Iterations_10k_2021_07_22_01_59_47/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10_Mode_2_Iterations_10k_2021_07_22_01_59_47/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_100_Mode_0_Iterations_10k_2021_07_21_23_52_49/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_100_Mode_0_Iterations_10k_2021_07_21_23_52_49/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_100_Mode_1_Iterations_10k_2021_07_22_00_32_40/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_100_Mode_1_Iterations_10k_2021_07_22_00_32_40/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_100_Mode_2_Iterations_10k_2021_07_22_02_05_11/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_100_Mode_2_Iterations_10k_2021_07_22_02_05_11/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_1000_Mode_0_Iterations_10k_2021_07_21_23_59_03/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_1000_Mode_0_Iterations_10k_2021_07_21_23_59_03/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_1000_Mode_1_Iterations_10k_2021_07_22_00_39_16/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_1000_Mode_1_Iterations_10k_2021_07_22_00_39_16/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_1000_Mode_2_Iterations_10k_2021_07_22_02_10_31/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_1000_Mode_2_Iterations_10k_2021_07_22_02_10_31/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10000_Mode_0_Iterations_10k_2021_07_22_00_06_41/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10000_Mode_0_Iterations_10k_2021_07_22_00_06_41/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10000_Mode_1_Iterations_10k_2021_07_22_01_00_16/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10000_Mode_1_Iterations_10k_2021_07_22_01_00_16/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10000_Mode_2_Iterations_10k_2021_07_22_02_16_08/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/three_linkage_sampling_comparisons/IK_3d_threelinkage_Samples_10000_Mode_2_Iterations_10k_2021_07_22_02_16_08/plots"
    }
]

for local_globals in local_globals_list :
    with open(compute_hist_grom_model_path) as f:
        code = compile(f.read(), compute_hist_grom_model_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = compute_hist_grom_model_path
        exec(code, globals())

'''

'''

# Experiment: fix histograms for ur5 experiments

local_globals_list = [
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10_Mode_0_Iterations_10k_2021_07_22_11_23_02/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10_Mode_0_Iterations_10k_2021_07_22_11_23_02/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10_Mode_1_Iterations_10k_2021_07_22_11_15_10/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10_Mode_1_Iterations_10k_2021_07_22_11_15_10/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10_Mode_2_Iterations_10k_2021_07_22_12_09_16/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10_Mode_2_Iterations_10k_2021_07_22_12_09_16/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_100_Mode_0_Iterations_10k_2021_07_22_11_31_18/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_100_Mode_0_Iterations_10k_2021_07_22_11_31_18/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_100_Mode_1_Iterations_10k_2021_07_22_11_05_04/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_100_Mode_1_Iterations_10k_2021_07_22_11_05_04/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_100_Mode_2_Iterations_10k_2021_07_22_12_27_27/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_100_Mode_2_Iterations_10k_2021_07_22_12_27_27/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_1000_Mode_0_Iterations_10k_2021_07_22_11_40_09/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_1000_Mode_0_Iterations_10k_2021_07_22_11_40_09/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_1000_Mode_1_Iterations_10k_2021_07_22_10_18_14/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_1000_Mode_1_Iterations_10k_2021_07_22_10_18_14/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_1000_Mode_2_Iterations_10k_2021_07_22_12_35_03/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_1000_Mode_2_Iterations_10k_2021_07_22_12_35_03/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10000_Mode_0_Iterations_10k_2021_07_22_11_49_03/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10000_Mode_0_Iterations_10k_2021_07_22_11_49_03/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10000_Mode_1_Iterations_10k_2021_07_22_02_35_37/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10000_Mode_1_Iterations_10k_2021_07_22_02_35_37/plots"
    },
    {
        "exp_DIR_MODEL": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10000_Mode_2_Iterations_10k_2021_07_22_12_42_51/model/nn_model_full",
        "exp_DIR_PATH_IMG": "D:/polybox_folder/master_thesis/experiments/main_experiments/ur5_sampling_comparisons/IK_3d_UR5_Samples_10000_Mode_2_Iterations_10k_2021_07_22_12_42_51/plots"
    }
]

for local_globals in local_globals_list :
    with open(compute_hist_grom_model_path) as f:
        code = compile(f.read(), compute_hist_grom_model_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = compute_hist_grom_model_path
        exec(code, globals())

'''

'''
# Experiment: The Activation Function Matters Too

local_globals_list = [
    {
        "exp_activation_function": "cos"
    },
    {
        "exp_activation_function": "sin"
    },
    {
        "exp_activation_function": "mish"
    },
    {
        "exp_activation_function": "sigmoid"
    },
    {
        "exp_activation_function": "tanh"
    },
    {
        "exp_activation_function": "tanhshrink"
    },
    {
        "exp_activation_function": "relu"
    },
    {
        "exp_activation_function": "leakyrelu"
    }
]

for local_globals in local_globals_list :
    with open(benchmark_path) as f:
        code = compile(f.read(), benchmark_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = benchmark_path
        globals()["exp_SAMPLING_MODE"] = 0
        globals()["exp_IS_TWOLINKAGE_CONSTRAINED"] = False
        globals()["exp_IS_MODE_2_ABLATION"] = False
        globals()["exp_N_SAMPLES_TRAIN"] = 1000
        globals()["exp_N_ITERATIONS"] = 100000
        exec(code, globals())

'''