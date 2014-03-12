import sys
import os
import subprocess
import pstats
#import gprof2dot

def make_graph(profile, output_png_file, node_thresh=0.5):
    proc_graph = subprocess.Popen(["./gprof2dot.py", "--skew", "0.5", "-n", "{:f}".format(node_thresh), 
                                   "-f", "pstats", profile], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


    # the directed graph is produced by proc_graph.stdout
    proc_dot = subprocess.Popen(["dot", "-Tpng", "-o", output_png_file], 
                                stdin = proc_graph.stdout, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    stdout, stderr = proc_dot.communicate()
    
profile_path = "."
profile_root = "prof."

profile_files = [fn for fn in os.listdir(profile_path) if fn.startswith(profile_root)];

first_profile = True

individual_core_profile = False


if len(profile_files) > 0:
    for i_prof, profile in enumerate(profile_files):
        print("Opening profile: ", profile)
        stats = pstats.Stats(profile)

        graph_image = "code_profile.{:d}.png".format(i_prof)

        if individual_core_profile:
            make_graph(profile, graph_image)
    
            stats.strip_dirs().sort_stats('tottime').print_stats(10)
            
        if first_profile:
            stats_pdf = {}
            for i, data in enumerate(stats.strip_dirs().sort_stats('tottime').stats.items()):
                func, (cc, nc, tt, ct, callers) = data


            first_profile = False
        else:
            total_stats.add(profile)


    print(80*"*")
    total_stats.strip_dirs().sort_stats('tottime').print_stats(10)
    total_stats.dump_stats("full_profile")

else:
    total_stats = pstats.Stats("full_profile")


graph_image = "full_code_profile.png"

make_graph("full_profile", graph_image)

threshhold_image = "above_5_percent.png"
make_graph("full_profile", threshhold_image, node_thresh=5)
