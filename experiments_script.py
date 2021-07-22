#!/bin/python3

import pathlib

benchmark_path = pathlib.Path(pathlib.Path(__file__).parent.resolve(), "benchmark.py")

glob_loc_list = [
    {
        "exp_SAMPLING_MODE": 0,
        "exp_IS_TWOLINKAGE_CONSTRAINED": False,
        "exp_IS_MODE_2_ABLATION": False,
        "exp_N_SAMPLES_TRAIN": 1000,
        "exp_N_ITERATIONS": 10000,
        "__file__": benchmark_path
    },
    {
        "exp_SAMPLING_MODE": 0,
        "exp_IS_TWOLINKAGE_CONSTRAINED": True,
        "exp_IS_MODE_2_ABLATION": False,
        "exp_N_SAMPLES_TRAIN": 1000,
        "exp_N_ITERATIONS": 10000,
        "__file__": benchmark_path
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_IS_TWOLINKAGE_CONSTRAINED": False,
        "exp_IS_MODE_2_ABLATION": False,
        "exp_N_SAMPLES_TRAIN": 1000,
        "exp_N_ITERATIONS": 10000,
        "__file__": benchmark_path
    },
    {
        "exp_SAMPLING_MODE": 1,
        "exp_IS_TWOLINKAGE_CONSTRAINED": True,
        "exp_IS_MODE_2_ABLATION": False,
        "exp_N_SAMPLES_TRAIN": 1000,
        "exp_N_ITERATIONS": 10000,
        "__file__": benchmark_path
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_IS_TWOLINKAGE_CONSTRAINED": False,
        "exp_IS_MODE_2_ABLATION": False,
        "exp_N_SAMPLES_TRAIN": 1000,
        "exp_N_ITERATIONS": 10000,
        "__file__": benchmark_path
    },
    {
        "exp_SAMPLING_MODE": 2,
        "exp_IS_TWOLINKAGE_CONSTRAINED": True,
        "exp_IS_MODE_2_ABLATION": False,
        "exp_N_SAMPLES_TRAIN": 1000,
        "exp_N_ITERATIONS": 10000,
        "__file__": benchmark_path
    },
]

for globals in glob_loc_list :
    with open(benchmark_path) as f:
        code = compile(f.read(), benchmark_path, "exec")
        exec(code, globals)