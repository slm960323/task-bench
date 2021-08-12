from __future__ import absolute_import, division, print_function

import core
from mpi4py import MPI
import sys
import time
import numpy as np


comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()
# start_time = None
# print(core)

def gen_rank_point_map(graph):
    rank_by_point = [0] * graph.max_width
    tag_bits_by_point = [0] * graph.max_width
    for r in range(n_ranks):
        first_point = r * graph.max_width / n_ranks
        last_point = (r + 1) * graph.max_width / n_ranks -1
        first_point = int(first_point)
        last_point = int(last_point)
        for p in range(first_point, last_point+1):
            rank_by_point[p] = r
            tag_bits_by_point[p] = p - first_point

    return rank_by_point, tag_bits_by_point

def execute_task_graph(graph):
    first_point = rank * graph.max_width / n_ranks
    last_point = (rank + 1) * graph.max_width / n_ranks -1
    first_point = int(first_point)
    last_point = int(last_point)
    n_points = last_point - first_point +1

    scratch_bytes = graph.scratch_bytes_per_task
    rank_by_point, tag_bits_by_point = gen_rank_point_map(graph)
    requests = []
    n_deps_map = {}


    ## Calculate max_deps
    max_deps = 0
    # print("core.c.task_graph_max_dependence_sets(graph) = ", core.c.task_graph_max_dependence_sets(graph))
    for dset in range(core.c.task_graph_max_dependence_sets(graph)):
        for p in range(first_point, last_point+1):
            dep_iter = core.c.task_graph_dependencies(graph, dset, p)
            deps = 0
            # print("p = {} dep_iter = {}; interval_list_num_intervals = {}".format(p, dep_iter, core.c.interval_list_num_intervals(dep_iter)))
            for i in range(0, core.c.interval_list_num_intervals(dep_iter)):
                interval = core.c.interval_list_interval(dep_iter, i)
                # print("interval = {}; ({}, {})".format(interval, interval.start, interval.end))
                deps += interval.end - interval.start + 1
            n_deps_map[p] = deps
            max_deps = max(max_deps, deps)
            # print("max_deps = ", max_deps)


    ## Create inputs
    input_bytes_per_task = graph.output_bytes_per_task
    inputs = [0] * n_points

    outputs = [np.zeros(input_bytes_per_task, dtype=np.ubyte)] * n_points

    ## Generate dependency and reverse_dependency maps
    dependencies = [None] * core.c.task_graph_max_dependence_sets(graph)
    reverse_dependencies = [None] * core.c.task_graph_max_dependence_sets(graph)
    for dset in range(core.c.task_graph_max_dependence_sets(graph)):
        dependencies[dset] = [None] * n_points
        reverse_dependencies[dset] = [None] * n_points
        # #print("len(dependencies[dest]) = {}".format(len(dependencies[dset])))

        for point in range(first_point, last_point+1):
            point_index = point - first_point
            dependencies[dset][point_index] = core.c.task_graph_dependencies(graph, dset, point)
            for i in range(0, core.c.interval_list_num_intervals(dependencies[dset][point_index])):
                    interval = core.c.interval_list_interval(dependencies[dset][point_index], i)
                    # print("Point {} dest {}  dep: ({}, {})".format(point, dset, interval.start, interval.end))
            reverse_dependencies[dset][point_index] = core.c.task_graph_reverse_dependencies(graph, dset, point)
            for i in range(0, core.c.interval_list_num_intervals(reverse_dependencies[dset][point_index])):
                    interval = core.c.interval_list_interval(reverse_dependencies[dset][point_index], i)
                    # print("Point {} dest {}  rev_dep: ({}, {})".format(point, dset, interval.start, interval.end))


    start_time = time.perf_counter()
    ## Run timestep
    for timestep in range(graph.timesteps):
        offset = core.c.task_graph_offset_at_timestep(graph, timestep)
        width  = core.c.task_graph_width_at_timestep(graph, timestep)
        

        last_offset = core.c.task_graph_offset_at_timestep(graph, timestep-1)
        last_width = core.c.task_graph_width_at_timestep(graph, timestep-1)

        dset = core.c.task_graph_dependence_set_at_timestep(graph, timestep)
        # #print("dset = {}".format(dset))
        deps = dependencies[dset]
        rev_deps = reverse_dependencies[dset]

        for point in range(first_point, last_point+1):
            #print("\n*** Point {} in [{}, {})".format(point, first_point, last_point+1))
            point_index = point - first_point
            point_deps = deps[point_index]
            point_rev_deps = rev_deps[point_index]
            inputs[point_index] = []
            # #print("point_deps = {}".format(point_deps))
            
            # Send
            if (point >= last_offset and point < last_offset + last_width):
                for i in range(0, core.c.interval_list_num_intervals(point_rev_deps)):
                    interval = core.c.interval_list_interval(point_deps, i)
                    # print("Point {} rev_deps_interval: ({}, {})".format(point, interval.start, interval.end))
                    for dep in range(interval.start, interval.end + 1):
                        # print("\tpoint {} rev_dep = {} ".format(point, dep))
                        if (dep < offset or dep >= offset + width or (first_point <= dep and dep <= last_point)):
                            # print("\t\tcontinue")
                            continue
                        
                        p_from = tag_bits_by_point[point]
                        p_to = tag_bits_by_point[dep]
                        tag = (p_from << 8) | p_to
                        # print("\t\tSend:: from = {}; to = {};  dest = {}; tag = {}; data = {}".format(p_from, p_to, rank_by_point[dep], tag, outputs[point_index]))
                        req = comm.Isend(outputs[point_index], dest=rank_by_point[dep], tag=tag)
                        requests.append(req)


            # Receive
            if (point >= offset and point < offset + width):
                for i in range(0, core.c.interval_list_num_intervals(point_deps)):
                    interval = core.c.interval_list_interval(point_deps, i)

                    for dep in range(interval.start, interval.end + 1):
                        if (dep < last_offset or dep >= last_offset + last_width):
                            continue

                        if (first_point <= dep and dep <= last_point):
                            output = outputs[dep - first_point]
                            inputs[point_index].append(np.copy(output))

                        else:
                            p_from = tag_bits_by_point[dep]
                            p_to = tag_bits_by_point[point]
                            tag = (p_from << 8) | p_to
                            inputs[point_index].append(np.empty(input_bytes_per_task, dtype=np.ubyte))
                            req = comm.Irecv([inputs[point_index][-1], MPI.BYTE], source=rank_by_point[dep], tag=tag)
                            requests.append(req)


        status = [MPI.Status() for i in range(len(requests))]
        MPI.Request.Waitall(requests, status)
        requests = []

        comm.Barrier()

        #print("n_deps_map = {}".format(n_deps_map))

        # print("before inputs = {}".format(inputs))

        for point in range(max(first_point, offset), min(last_point, offset + width -1)+1): 
            point_index = point - first_point
            point_inputs = np.copy(np.array(inputs[point_index]))

            # size = [i.shape[0] for i in inputs[point_index]]
            # print("Final:: point = {} point_index = {}; inputs = {}".format(point, point_index, point_inputs))
            if timestep == 0:
                outputs[point_index] = core.execute_point_impl(graph, timestep, point, None, 0, *point_inputs)
            else:
                outputs[point_index] = core.execute_point_impl(graph, timestep, point, None, n_deps_map[point], *np.array(inputs[point_index]))
                pass
            # print("Final:: point = {}; point_index = {};  output = {}\noutputs = {}".format(point, point_index, outputs[point_index], outputs))

        # print("after inputs = {}\n\n\n".format(inputs))

    comm.Barrier()
    return start_time


def execute_task_bench():
    app = core.app_create(sys.argv)
    if rank == 0:
        core.c.app_display(app)
    task_graphs = core.app_task_graphs(app)
    start_time = time.perf_counter()
    # results = []
    for task_graph in task_graphs:
        start_time = execute_task_graph(task_graph)
    for task_graph in task_graphs:
        start_time = execute_task_graph(task_graph)
    stop_time = time.perf_counter()
    elapsed_time = stop_time - start_time

    if rank == 0:
        core.c.app_report_timing(app, elapsed_time)


if __name__ == "__main__":
    execute_task_bench()