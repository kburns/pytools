import pstats_ben as pstats
import sys
import os
import subprocess
#import gprof2dot

def make_graph(profile, output_png_file, node_thresh=0):
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

loop_over_all = False
if loop_over_all:
    for i_prof, profile in enumerate(profile_files):
        print("Opening profile: ", profile)
        stats = pstats.Stats(profile)

        graph_image = "code_profile.{:d}.png".format(i_prof)

        if individual_core_profile:
            make_graph(profile, graph_image)
    
        stats.strip_dirs().sort_stats('tottime').print_stats(10)

        if first_profile:
            total_stats = pstats.Stats(profile)
            first_profile = False
        else:
            total_stats.add(profile)

else:
    # fast read-in version
    total_stats = pstats.Stats(profile_files[0])

    total_stats.add(*profile_files[1:])

    nightmare = True
    if nightmare:
        #for func, stats in total_stats.sort_stats('tottime').stats.items():
        #    print(total_stats.stats[func])

        # requires modified pstats
        avg_stats = pstats.Stats(profile_files[0])
        
        avg_stats.average(*profile_files[1:])
        
    
        # requires modified pstats
        mean_stats = pstats.Stats(profile_files[0])
        
        mean_stats.mean(*profile_files[1:])
    
        # requires modified pstats
        stddev_stats = pstats.Stats(profile_files[0])
        
        stddev_stats.stddev(*profile_files[1:])
    
    
print(80*"*")
total_stats.strip_dirs().sort_stats('tottime').print_stats(10)
total_stats.dump_stats("full_profile")

if nightmare:
    avg_stats.strip_dirs().sort_stats('tottime').print_stats(10)

    mean_stats.strip_dirs().sort_stats('tottime').print_stats(10)

    stddev_stats.strip_dirs().sort_stats('tottime').print_stats(10)


graph_image = "full_code_profile.png"

make_graph("full_profile", graph_image)

threshhold_image = "above_5_percent.png"
make_graph("full_profile", threshhold_image, node_thresh=0.05)
