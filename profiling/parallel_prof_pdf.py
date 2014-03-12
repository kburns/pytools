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

def make_pdf(stats_pdf_dict, total_time, label='', N_profiles=20):

    sorted_list = sorted(stats_pdf_dict.items(), key=lambda data_i: np.max(data_i[1]), reverse=True)

    routine_text = "top {:d} routines for {:s}".format(N_profiles, label)
    print()
    print("{:80s}".format(routine_text),"     min    mean     max   (mean%total)")
    print(120*"-")
    for i_fig, (func, data_list) in enumerate(sorted_list):
        data = np.array(data_list)
        N_data = data.shape[0]

        if N_data > 200:
            N_bins = 100
        else:
            N_bins = N_data

        if func[0] == '~':
            title_string = func[2]
        else:
            title_string = "{:s}:{:d}:{:s}".format(*func)

        def percent_time(sub_time):
            sub_string = "{:.2g}%".format(100*sub_time/total_time)
            return sub_string

        timing_data_string = "{:6.2g} |{:6.2g} |{:6.2g}  ({:s})".format(np.min(data), np.mean(data), np.max(data), percent_time(np.mean(data)))

        print("{:80s} = {:s}".format(title_string, timing_data_string))

        timing_data_string = "min {:s} | {:s} | {:s} max".format(percent_time(np.min(data)), percent_time(np.mean(data)), percent_time(np.max(data)))

        title_string += "\n{:s}".format(timing_data_string)
        
        hist_values, bin_edges = np.histogram(data, bins=N_bins)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)

        ax1.plot(hist_values, bin_edges[1:])
            
        ax2 = fig.add_subplot(1,2,2)
        ax2.bar(np.arange(N_data), data)

        ax1.set_ylim(0, 1.1*np.max(data))
        ax2.set_ylim(0, 1.1*np.max(data))


        fig.suptitle(title_string)
        fig.savefig(label+'_{:06d}.png'.format(i_fig+1))
        plt.close(fig)

        if i_fig+1 == N_profiles:
            break
    
def read_prof_files(profile_files, 
                    individual_core_profile=False, include_dir_info=False, 
                    database_file='profile_data.db', stats_file='full_profile.stats'):
    
    first_profile = True
    
    stats_pdf_tt = {}
    stats_pdf_ct = {}
    for i_prof, profile in enumerate(profile_files):
        print("Opening profile: ", profile)
        stats = pstats.Stats(profile)

        if not include_dir_info:
            stats = stats.strip_dirs()

        if individual_core_profile:
            graph_image = "code_profile.{:d}.png".format(i_prof)
            make_graph(profile, graph_image)
    
            stats.sort_stats('tottime').print_stats(10)
            
        if first_profile:
            first_profile = False
            total_stats = stats
        else:
            total_stats.add(stats)

        for data in stats.stats.items():
            func, (cc, nc, tt, ct, callers) = data
            if func in stats_pdf_tt:
                stats_pdf_tt[func].append(tt)
                stats_pdf_ct[func].append(ct)
            else:
                stats_pdf_tt[func] = [tt]
                stats_pdf_ct[func] = [ct]
            
    N_cpu = i_prof+1

    print(80*"*")
    total_time = total_stats.total_tt/N_cpu

    print("total run time: {:g} sec per core".format(total_time))
    print()
    
    shelf = shelve.open(database_file, flag='n')
    shelf['stats_pdf_tt'] = stats_pdf_tt
    shelf['stats_pdf_ct'] = stats_pdf_ct
    shelf['total_time'] = total_time
    shelf['N_cpu'] = N_cpu

    total_stats.dump_stats(stats_file)

    return total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu, total_time

def read_database(database_file='profile_data.db', stats_file='full_profile.stats'):
    shelf = shelve.open(database_file, flag='r')
    stats_pdf_tt = shelf['stats_pdf_tt']
    stats_pdf_ct = shelf['stats_pdf_ct']
    total_time = shelf['total_time']
    N_cpu = shelf['N_cpu']

    total_stats = pstats.Stats(stats_file)

    return total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu, total_time




database_file = 'profile_data.db'
stats_file = 'full_profile.stats'

if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        profile_path = sys.argv[1]
        profile_root = sys.argv[2]
    else:
        profile_path = "."
        profile_root = sys.argv[1]
        
    profile_files = [fn for fn in os.listdir(profile_path) if fn.startswith(profile_root)];
    total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu, total_time = read_prof_files(profile_files, database_file=database_file, stats_file=stats_file, include_dir_info=False)
else:
    print(80*'*')
    print("restoring pre-generated databases")
    total_stats, stats_pdf_tt, stats_pdf_ct, N_cpu, total_time = read_database(database_file=database_file, stats_file=stats_file)

    
total_stats.strip_dirs().sort_stats('tottime').print_stats(10)

print("creating PDFs over {:d} cpu".format(N_cpu))
make_pdf(stats_pdf_tt, total_time, label="tt")

graph_image = "full_code_profile.png"

make_graph(stats_file, graph_image)

threshhold_image = "above_5_percent.png"
make_graph(stats_file, threshhold_image, node_thresh=5)
