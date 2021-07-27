#!/bin/python3

import pathlib

benchmark_path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), "benchmark.py")

# Experiment: fix histograms
local_globals_list = [
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
        "exp_N_SAMPLES_TRAIN": 1000
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_N_SAMPLES_TRAIN": 1000
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_N_SAMPLES_TRAIN": 1000
    },
    {
        "exp_SAMPLING_MODE": 0,
        "exp_N_SAMPLES_TRAIN": 10000
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_N_SAMPLES_TRAIN": 10000
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_N_SAMPLES_TRAIN": 10000
    }
]

for local_globals in local_globals_list :
    with open(benchmark_path) as f:
        code = compile(f.read(), benchmark_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = benchmark_path
        globals()["exp_N_ITERATIONS"] = 10000
        globals()["exp_IS_MODE_2_ABLATION"] = False
        globals()["exp_IS_TWOLINKAGE_CONSTRAINED"] = False
        globals()["exp_SEED_DICT"] = {
            "bench_random_seed": 4532,
            "bench_numpy_random_seed": 4542,
            "bench_torch_random_seed": 242,
            "ik_random_seed": 2521,
            "ik_numpy_random_seed": 2781,
            "ik_torch_random_seed": 261,
            "helper_random_seed": 151,
            "helper_numpy_random_seed": 1611,
            "helper_torch_random_seed": 171
        }
        exec(code, globals())


'''
# Experiment: The RNG Matters

local_globals_list = [
    {
        "exp_SAMPLING_MODE": 0,
    },
    {
        "exp_SAMPLING_MODE": 1,
    },
    {
        "exp_SAMPLING_MODE": 2,
    }
]

for local_globals in local_globals_list :
    with open(benchmark_path) as f:
        code = compile(f.read(), benchmark_path, "exec")
        globals().update(local_globals)
        globals()["__file__"] = benchmark_path
        globals()["exp_N_SAMPLES_TRAIN"] = 1000
        globals()["exp_N_ITERATIONS"] = 10000
        globals()["exp_IS_MODE_2_ABLATION"] = False
        globals()["exp_IS_TWOLINKAGE_CONSTRAINED"] = False
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