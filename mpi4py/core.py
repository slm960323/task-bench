from __future__ import absolute_import, division, print_function

import sys
import time
# import task_bench_core as core
import cffi
import os
import subprocess
import numpy as np

SENDING = False
RECEIVING = True

root_dir = os.path.dirname(os.path.dirname(__file__))
core_header = subprocess.check_output(
    [
        "gcc", "-D", "__attribute__(x)=", "-E", "-P",
        os.path.join(root_dir, "core/core_c.h")
    ]).decode("utf-8")
ffi = cffi.FFI()
ffi.cdef(core_header)
c = ffi.dlopen("libcore.so")

def app_create(args):
    c_args = []
    c_argv = ffi.new("char *[]", len(args) + 1)
    for i, arg in enumerate(args):
        c_args.append(ffi.new("char []", arg.encode('utf-8')))
        c_argv[i] = c_args[-1]
    c_argv[len(args)] = ffi.NULL

    app = c.app_create(len(args), c_argv)
    # c.app_display(app)
    return app

def app_task_graphs(app):
    result = []
    graphs = c.app_task_graphs(app)

    for i in range(c.task_graph_list_num_task_graphs(graphs)):
        result.append(c.task_graph_list_task_graph(graphs, i))

    return result

def init_scratch_direct(scratch_bytes):
    scratch = np.empty(scratch_bytes, dtype=np.ubyte)
    scratch_ptr = ffi.cast("char *", scratch.ctypes.data)
    c.task_graph_prepare_scratch(scratch_ptr, scratch_bytes)
    return scratch

def execute_point_impl(graph, timestep, point, scratch, input_len, *inputs):
    input_ptrs = ffi.new(
        "char *[]", [ffi.cast("char *", i.ctypes.data) for i in inputs])
    input_sizes = ffi.new("size_t []", [i.shape[0] for i in inputs])
    sizes = [i.shape[0] for i in inputs]
    # print("!!!EXE:: sizes = {} {}".format(len(inputs)))

    output = np.zeros(graph.output_bytes_per_task, dtype=np.ubyte)
    output_ptr = ffi.cast("char *", output.ctypes.data)

    if scratch is not None:
        scratch_ptr = ffi.cast("char *", scratch.ctypes.data)
        scratch_size = scratch.shape[0]
    else:
        scratch_ptr = ffi.NULL
        scratch_size = 0

    #print("timestep {}; point {}; output.shape(0) {}; input_sizes {}; len(inputs) {}; scratch_size {}".format(
    #     timestep, point, output.shape[0], sizes, input_len, scratch_size
    # ))

    c.task_graph_execute_point_scratch(
        graph, timestep, point, output_ptr, output.shape[0], input_ptrs,
        input_sizes, input_len, scratch_ptr, scratch_size)

    return output

