import sys
import os
import subprocess
import pstats
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
import shelve
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

def make_pdf(stats_pdf_dict, label='', N_profiles=20):
    print("creating PDFs over {:d} cpu".format(N_cpu))

    sorted_list = sorted(stats_pdf_dict.items(), key=lambda data_i: np.mean(data_i[1]), reverse=True)

    for i_fig, (func, data_list) in enumerate(sorted_list):
        data = np.array(data_list)
        N_data = data.shape[0]
        
        hist_values, bin_edges = np.histogram(data, bins=N_cpu)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(bin_edges[1:], hist_values)
        print(func, )
        ax1.set_title(func[0]+func[2])
        ax2 = fig.add_subplot(1,2,2)
        ax2.bar(np.arange(N_data), data)
        fig.savefig(label+'_{:06d}.png'.format(i_fig+1))
        plt.close(fig)

        if i_fig == N_profiles:
            break
    
def read_prof_files(profile_files, individual_core_profile=False, database_file='profile_data.db'):
    first_profile = True
    
    stats_pdf_tt = {}
    stats_pdf_ct = {}
    for i_prof, profile in enumerate(profile_files):
        print("Opening profile: ", profile)
        stats = pstats.Stats(profile)

        if individual_core_profile:
            graph_image = "code_profile.{:d}.png".format(i_prof)
            make_graph(profile, graph_image)
    
            stats.strip_dirs().sort_stats('tottime').print_stats(10)
            
        if first_profile:
            first_profile = False
            total_stats = stats
        else:
            total_stats.add(stats)
        for data in stats.strip_dirs().stats.items():
            func, (cc, nc, tt, ct, callers) = data
            if func in stats_pdf_tt:
                stats_pdf_tt[func].append(tt)
                stats_pdf_ct[func].append(ct)
            else:
                stats_pdf_tt[func] = [tt]
                stats_pdf_ct[func] = [ct]
            
    N_cpu = i_prof+1
            
    print(80*"*")
    total_stats.dump_stats("full_profile")

    shelf = shelve.open(database_file, flag='n')
    shelf['total_stats'] = total_stats
    shelf['stats_pdf_tt'] = stats_pdf_tt
    shelf['stats_pdf_ct'] = stats_pdf_ct
    shelf['N_cpu'] = N_cpu
    
    return total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu

def read_database(database_file):
    shelf = shelve.open(database_file, flag='r')
    total_stats = shelf['total_stats']
    stats_pdf_tt = shelf['stats_pdf_tt']
    stats_pdf_ct = shelf['stats_pdf_ct']
    N_cpu = shelf['N_cpu']
    
    return, total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu



profile_path = "."
profile_root = "prof."

profile_files = [fn for fn in os.listdir(profile_path) if fn.startswith(profile_root)];
    

if len(profile_files) > 0:
    total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu = read_prof_files(profile_files, database_file='profile_data.db')
else:
    total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu = read_database('profile_data.db')

    #total_stats = pstats.Stats("full_profile")

    
total_stats.strip_dirs().sort_stats('tottime').print_stats(10)

make_pdf(stats_pdf_tt, label="tt")

graph_image = "full_code_profile.png"

make_graph("full_profile", graph_image)

threshhold_image = "above_5_percent.png"
make_graph("full_profile", threshhold_image, node_thresh=5)
