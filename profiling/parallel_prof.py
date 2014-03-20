import sys
import os
import subprocess
import pstats
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import operator
import shelve
import brewer2mpl

#import gprof2dot

def set_plot_defaults():
    # Set up some better defaults for matplotlib
    # http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_03_statistical_graphs.ipynb
    #colorbrewer2 Dark2 qualitative color table
    bar_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
    bar_colors = brewer2mpl.get_map('Set3', 'Qualitative', 12).mpl_colors
    bar_colors = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors

    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    rcParams['axes.color_cycle'] = bar_colors
    rcParams['lines.linewidth'] = 2
    rcParams['axes.facecolor'] = 'white'
    rcParams['font.size'] = 14
    rcParams['patch.edgecolor'] = 'white'
    rcParams['patch.facecolor'] = bar_colors[0]
    rcParams['font.family'] = 'StixGeneral'

def make_graph(profile, output_png_file, node_thresh=0.5):
    proc_graph = subprocess.Popen(["./gprof2dot.py", "--skew", "0.5", "-n", "{:f}".format(node_thresh), 
                                   "-f", "pstats", profile], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


    # the directed graph is produced by proc_graph.stdout
    proc_dot = subprocess.Popen(["dot", "-Tpng", "-o", output_png_file], 
                                stdin = proc_graph.stdout, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    stdout, stderr = proc_dot.communicate()


def sort_dict(dict_to_sort):
    sorted_list = sorted(dict_to_sort.items(), key=lambda data_i: test_criteria(data_i[1]), reverse=True)
    return sorted_list

def test_criteria(data):
    return np.max(data)

def clean_display(ax):
    # from http://nbviewer.ipython.org/gist/anonymous/5357268
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')


def make_pdf(stats_pdf_dict, total_time, label='', N_profiles=20, thresh=0.01):

    sorted_list = sort_dict(stats_pdf_dict)

    composite_data_set = []
    composite_label = []
    composite_key_label = []

    fig_stacked = plt.figure()
    ax_stacked = fig_stacked.add_subplot(1,1,1)

    i_mpi_list = []
    i_fftw_list = []
    i_fft_list = []
    fft_type_list = ["ifft", "_dct", "rfft"]
    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break
        
        if "gssv" in func[2]:
            print("found sparse solve call:",func[2], " at ", i_sort) 
            i_gssv = i_sort
            
        if "mpi4py.MPI" in func[2]:
            print("found MPI call:",func[2], " at ", i_sort) 
            i_mpi_list.append(i_sort)

        if "fftw.fftw_wrappers.Transpose" in func[2]:
            print("found fftw transpose call:",func[2], " at ", i_sort) 
            i_fftw_list.append(i_sort)

        if any(fft_type in func[2] for fft_type in fft_type_list):
            print("found fft call:",func[2], " at ", i_sort) 
            i_fft_list.append(i_sort)
        
    # bubble sparse solve to the top
    sorted_list.insert(0,sorted_list.pop(i_gssv))
    last_insert = 0
    resort_lists = [i_mpi_list, i_fftw_list, i_fft_list]
    for i_resort_list in resort_lists:
        for i_resort in i_resort_list:
            sorted_list.insert(last_insert+1,sorted_list.pop(i_resort))
            print("moved entry {:d}->{:d}".format(i_resort, last_insert+1))
            last_insert += 1
                    
    routine_text = "top {:d} routines for {:s}".format(N_profiles, label)
    print()
    print("{:80s}".format(routine_text),"     min      mean       max   (mean%total)")
    print(120*"-")
    for i_fig, (func, data_list) in enumerate(sorted_list):
        data = np.array(data_list)
        N_data = data.shape[0]

        if i_fig+1 == N_profiles or (i_fig > last_insert and test_criteria(data)/total_time < thresh):
            break

        if i_fig == 0:
            previous_data = np.zeros_like(data)

        N_missing = previous_data.size - data.size
        
        if N_missing != 0:
            print("missing {:d} values; setting to zero".format(N_missing))
            for i in range(N_missing):
                data_list.insert(N_missing*(i+1)-1, 0)
            data = np.array(data_list)
            N_data = data.shape[0]

        if func[0] == '~':
            title_string = func[2]
        else:
            title_string = "{:s}:{:d}:{:s}".format(*func)

        def percent_time(sub_time):
            sub_string = "{:.2g}%".format(100*sub_time/total_time)
            return sub_string

        timing_data_string = "{:8.2g} |{:8.2g} |{:8.2g}  ({:s})".format(np.min(data), np.mean(data), np.max(data), percent_time(np.mean(data)))

        print("{:80s} = {:s}".format(title_string, timing_data_string))

        timing_data_string = "min {:s} | {:s} | {:s} max".format(percent_time(np.min(data)), percent_time(np.mean(data)), percent_time(np.max(data)))

        title_string += "\n{:s}".format(timing_data_string)

        key_label = "{:s} {:s}".format(percent_time(np.mean(data)),func[2])
        short_label = "{:s}".format(percent_time(np.mean(data)))
        
        composite_data_set.append([data])
        composite_label.append(short_label)
        composite_key_label.append(key_label)

            
        if N_data > 200:
            N_bins = 100
            logscale = True
        else:
            N_bins = N_data/4
            logscale = False

        q_color = next(ax_stacked._get_lines.color_cycle)
        
        fig = plt.figure()

        # pdf plot over many cores
        ax1 = fig.add_subplot(1,2,1)

        #hist_values, bin_edges = np.histogram(data, bins=N_bins)
        #ax1.barh(hist_values, bin_edges[1:])
        ax1.hist(data, bins=N_bins, orientation='horizontal', log=logscale, linewidth=0, color=q_color)
        ax1.set_xlabel("N cores/bin")
        ax1.set_ylabel("time (sec)")
        ax1.grid(axis = 'x', color ='white', linestyle='-')

        
        # bar plot for each core
        ax2 = fig.add_subplot(1,2,2)
        ax2.bar(np.arange(N_data), data, linewidth=0, width=1, color=q_color)
        ax2.set_xlim(-0.5, N_data+0.5)
        ax2.set_xlabel("core #")
        clean_display(ax2)
    
        ax2.grid(axis = 'y', color ='white', linestyle='-')

        # end include
        
        ax1.set_ylim(0, 1.1*np.max(data))
        ax2.set_ylim(0, 1.1*np.max(data))


        fig.suptitle(title_string)
        fig.savefig(label+'_{:06d}.png'.format(i_fig+1), dpi=200)
        plt.close(fig)
        
        ax_stacked.bar(np.arange(N_data), data, bottom=previous_data, label=short_label, linewidth=0,
                       width=1, color=q_color)
        previous_data += data

    clean_display(ax_stacked)
    ax_stacked.set_xlim(-0.5, N_data+0.5)
    ax_stacked.set_xlabel('core #')
    ax_stacked.set_ylabel('total time (sec)')
    ax_stacked.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    ax_stacked.set_title("per core timings for routines above {:g}% total time".format(thresh*100))
    ax_stacked.grid(axis = 'y', color ='white', linestyle='-')
    points_per_data = 10
    fig_x_size = 10
    fig_stacked.savefig(label+"_per_core_timings.png", dpi=max(200, N_data*points_per_data/fig_x_size))
    plt.close(fig_stacked)


    # pdf plot over many cores    
    fig_composite = plt.figure()
    ax_composite = fig_composite.add_subplot(1,1,1)

    n, bins, patches = ax_composite.hist(composite_data_set, bins=N_bins, orientation='vertical', log=logscale, linewidth=0, stacked=True,
                                         label=composite_label)

    clean_display(ax_composite)
    ax_composite.grid(axis = 'y', color ='white', linestyle='-')

    ax_composite.set_ylabel("N cores/bin")
    ax_composite.set_xlabel("total time (sec)")
    ax_composite.set_ylim(0, 1.1*np.max(composite_data_set))
    ax_composite.legend(loc='upper left', bbox_to_anchor=(1.,1.), fontsize=10)
    
    fig_composite.suptitle("composite PDF for routines above {:g}% total time".format(thresh*100))
    fig_composite.savefig(label+'_composite.png', dpi=200)
    plt.close(fig_composite)
    
    fig_key = plt.figure()
    plt.figlegend(patches, composite_key_label, 'center')
    #ax_key.legend(loc='center')
    fig_key.savefig(label+"_composite_key.png")
    plt.close(fig_key)

    



    
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
set_plot_defaults()

make_pdf(stats_pdf_tt, total_time, label="tt")


graph_image = "full_code_profile.png"

make_graph(stats_file, graph_image)

threshhold_image = "above_5_percent.png"
make_graph(stats_file, threshhold_image, node_thresh=5)

threshhold_image = "above_1_percent.png"
make_graph(stats_file, threshhold_image, node_thresh=1)
