#!/bin/python3

with open("somefile.py") as f:
    code = compile(f.read(), "somefile.py", 'exec')
    exec(code, global_vars, local_vars)