# master_thesis


## Requirements

Windows 10 64-bit
Python 3.9.6 64-bit
NumPy 1.19.5
Matplotlib 3.4.2
PyTorch 1.9.0+cu111

## How to start

Start file is benchmark.py
There are 3 possible experiments

1. Two-Linkage 2D IK
2. Three-Linkage 3D IK
3. UR5 3D IK

To choose between 2D and 3D, set the corresponding import file in benchmark.py.
To choose between Three-Linkage and UR5 in 3D, set the corresponding parameters in IK_3d_three_linkage.py

All important adjustable parameters are at the top of the files, including the helper.py file.
There is also a experiments_script.py file which can be used to automate series of experiments.