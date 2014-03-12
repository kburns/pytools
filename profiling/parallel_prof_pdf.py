import sys
import os
import subprocess
import pstats
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
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
    stats_pdf_tt = {}
    stats_pdf_ct = {}
    for i_prof, profile in enumerate(profile_files):
        print("Opening profile: ", profile)
        stats = pstats.Stats(profile)

        graph_image = "code_profile.{:d}.png".format(i_prof)

        if individual_core_profile:
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
    print("creating PDFs over {:d} cpu".format(N_cpu))
    dict_set = [("tt", stats_pdf_tt), ("ct", stats_pdf_ct)]
    for label, dict_now in dict_set:
        sorted_list = sorted(dict_now.items(), key=lambda data_i: np.mean(data_i[1]), reverse=True)
        i_fig = 1
        for func, data_list in sorted_list:
            data = np.array(data_list)
            #data = np.array(stats_pdf_tt[func])
            N_data = data.shape[0]
            
            hist_values, bin_edges = np.histogram(data, bins=N_cpu)
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax1.plot(bin_edges[1:], hist_values)
            print(func, )
            ax1.set_title(func[0]+func[2])
            ax2 = fig.add_subplot(1,2,2)
            ax2.bar(np.arange(N_data), data)
            fig.savefig(label+'_{:06d}.png'.format(i_fig))
            plt.close(fig)
            i_fig += 1
            if i_fig > 20:
                break
            
            
    print(80*"*")
    total_stats.strip_dirs().sort_stats('tottime').print_stats(10)
    total_stats.dump_stats("full_profile")

else:
    total_stats = pstats.Stats("full_profile")


graph_image = "full_code_profile.png"

make_graph("full_profile", graph_image)

threshhold_image = "above_5_percent.png"
make_graph("full_profile", threshhold_image, node_thresh=5)
