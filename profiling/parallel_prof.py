# import sys
# import os
# import subprocess
# import pstats
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import rcParams
# import matplotlib.pyplot as plt
# import operator
# import shelve
# import brewer2mpl

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
    exclude_list = ["load_dynamic", "__init__", "<frozen", "importlib"]

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break

        if "gssv" in func[2]:
            print("found sparse solve call:",func[2], " at ", i_sort)
            i_gssv = i_sort

        if "mpi4py.MPI" in func[2]:
            print("found MPI call:",func[2], " at ", i_sort)
            i_mpi_list.append(i_sort)

    # bubble sparse solve to the top
    sorted_list.insert(0,sorted_list.pop(i_gssv))
    last_insert = 0
    # insert MPI calls next
    for i_resort in i_mpi_list:
            sorted_list.insert(last_insert+1,sorted_list.pop(i_resort))
            print("moved entry {:d}->{:d}".format(i_resort, last_insert+1))
            last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if "fftw.fftw_wrappers.Transpose" in func[2]:
            print("found fftw transpose call:",func[2], " at ", i_sort)
            sorted_list.insert(last_insert+1,sorted_list.pop(i_sort))
            print("moved entry {:d}->{:d}".format(i_sort, last_insert+1))
            last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if any(fft_type in func[2] for fft_type in fft_type_list):
            print("found fft call:",func[2], " at ", i_sort)
            if i_sort < N_profiles:
                sorted_list.insert(last_insert+1,sorted_list.pop(i_sort))
                print("moved entry {:d}->{:d}".format(i_sort, last_insert+1))
                last_insert += 1

    for i_sort, (func, data_list) in enumerate(sorted_list):
        if i_sort+1 == N_profiles:
            break
        if any((exclude_type in func[0] or exclude_type in func[2]) for exclude_type in exclude_list):
            print("found excluded call:",func[2], " at ", i_sort, " ... popping.")
            sorted_list.pop(i_sort)


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



joined_filename = 'joined_stats.db'
summed_filename = 'summed_stats.prof'

def combine_profiles(directory, filenames, verbose=False):
    """Combine statistics from a collection of profiles."""

    import os
    import pstats
    import shelve
    from collections import defaultdict
    from contextlib import closing

    summed_stats = pstats.Stats()
    joined_primcalls = defaultdict(list)
    joined_totcalls = defaultdict(list)
    joined_tottime = defaultdict(list)
    joined_cumtime = defaultdict(list)

    if verbose:
        print("Combining profiles:")

    for i, filename in enumerate(filenames):
        if verbose:
            print("  {:s}".format(filename))

        stats = pstats.Stats(filename)
        stats.strip_dirs()
        summed_stats.add(stats)

        for funcstats in stats.stats.items():
            func, (primcalls, totcalls, tottime, cumtime, callers) = funcstats
            joined_primcalls[func].append(primcalls)
            joined_totcalls[func].append(totcalls)
            joined_tottime[func].append(tottime)
            joined_cumtime[func].append(cumtime)

    n_processes = len(filenames)
    average_runtime = summed_stats.total_tt / n_processes
    if verbose:
        print("  Average runtime: {:g} s".format(average_runtime))

    summed_stats.dump_stats(os.path.join(directory, summed_filename))

    with closing(shelve.open(os.path.join(directory, joined_filename), flag='n')) as shelf:
        shelf['primcalls'] = joined_primcalls
        shelf['totcalls'] = joined_totcalls
        shelf['tottime'] = joined_tottime
        shelf['cumtime'] = joined_cumtime
        shelf['average_runtime'] = average_runtime
        shelf['n_processes'] = n_processes


def read_database(directory):

    summed_stats = pstats.Stats(os.path.join(directory, summed_filename))

    with shelve.open(os.path.join(directory, joined_filename), flag='r') as shelf:
        primcalls = shelf['primcalls']
        totcalls = shelf['totcalls']
        tottime = shelf['tottime']
        cumtime = shelf['cumtime']
        average_runtime = shelf['average_runtime']
        n_processes = shelf['n_processes']

    return summed_stats, primcalls, totcalls, tottime, cumtime, average_runtime, n_processes


# print("creating PDFs over {:d} cpu".format(N_cpu))
# set_plot_defaults()

# make_pdf(stats_pdf_tt, total_time, label="tt")

# graph_image = "full_code_profile.png"

# make_graph(stats_file, graph_image)

# threshhold_image = "above_5_percent.png"
# make_graph(stats_file, threshhold_image, node_thresh=5)

# threshhold_image = "above_1_percent.png"
# make_graph(stats_file, threshhold_image, node_thresh=1)


if __name__ == "__main__":

    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(description="Analyze parallel python profiles.")
    parser.add_argument('command', choices=['process', 'plot'], help="Combine profiles into database, or plot database")
    parser.add_argument('directory', nargs='?', default='.', help="Directory containing profiles / database")
    parser.add_argument('pattern', nargs='?', default='proc_*.prof', help="Profile naming pattern (e.g. proc_*.prof)")
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    if args.command == 'process':
        pathname = os.path.join(args.directory, args.pattern)
        filenames = glob.glob(pathname)
        combine_profiles('.', filenames, verbose=args.verbose)
    elif args.command == 'plot':
        pass
    else:
        raise ValueError("Error parsing commands.")

