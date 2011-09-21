from __future__ import division
from kalderstam.util.filehandling import read_data_file, parse_data
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None #This makes matplotlib optional
    cm = None
import numpy as np

def show():
    '''
    Show plots which have been drawn but not displayed yet. Simply a call to matplotlib.pyplot's show()
    '''
    if plt:
        plt.show()

def kaplanmeier(data = None, time_column = None, event_column = None, output_column = None, time_array = None, event_array = None, output_array = None, threshold = None, even = False, show_plot = True):
    ''' Idea is to plot number of patients still alive on the y-axis,
    against the time on the x-axis. The time column specifies which
    axis is the time. Event column should be a binary value, indicating
    1 for an event and 0 for (right)censored. 
    Output column specifies the network output column
    and threshold the value(s) to divide the groups by (network output).
    If no threshold is specified, the set will be divided into 2 equally large
    sets, divided by the median.
    '''
    if (time_column is None and time_array is None or
        time_column is not None and time_array is not None):
        raise Exception("Both time and column can't be None/used")
    if (event_column is None and event_array is None or
        event_column is not None and event_array is not None):
        raise Exception("Both event and column can't be None/used")
    if (output_column is None and output_array is None or
        output_column is not None and output_array is not None):
        raise Exception("Both output and column can't be None/used")
    if (data is not None and time_column is None and event_column is None and output_column is None):
        raise Exception("Don't know what to do with data without column information..")

    if plt:

        if time_column is not None:
            time_array = data[:, time_column]
        if event_column is not None:
            event_array = data[:, event_column]
        if output_column is not None:
            output_array = data[:, output_column]


        if threshold is None:
            threshold = np.median(output_array)
        try:
            list(threshold)
        except TypeError:
            threshold = [threshold] #Make it a list
        #Make sure it's sorted lowest to highest
        threshold = sorted(threshold)

        #Divide set
        times = [[] for _ in xrange(len(threshold) + 1)]
        alive = [[] for _ in xrange(len(threshold) + 1)]

        for time, event, value in zip(time_array, event_array, output_array):
            for i in xrange(len(times)):
                if i < len(threshold) and value < threshold[i]:
                    times[i].append((time, event))
                    break
                elif i == len(threshold):
                    times[i].append((time, event))

        #Special case when median actually equals a lot of values, will happen when you have many tail-censored
        if even:
            remove_list = [[] for _ in range(len(times))]
            for i in xrange(1, len(times)):
                for j in xrange(len(times[i])):
                    if times[i][j][0] == threshold[i - 1] and len(times[i]) >= len(times[i - 1]) + 2:
                        times[i - 1].append(times[i][j])
                        remove_list[i].append(j)
            for i in xrange(len(remove_list)):
                for j in sorted(remove_list[i], reverse = True):
                    times[i].pop(j)

        for i in xrange(len(times)):
            times[i] = sorted(times[i], key = lambda x: x[0])

        #Now make list of all time indices, this is just a convenience for plotting points
        all_times = []
        for time in times:
            all_times += time
        all_times = sorted(all_times, key = lambda x: x[0])
        #all_times = sorted(times[0] + times[1], key = lambda x: x[0])

        ticklabels = ["{0}y".format(year) for year in xrange(int(all_times[-1][0]) + 1)]

        #Count how many are alive at each time
        for i in sorted(xrange(len(times)), reverse = True): #Take best survivors first
            prev = -1
            for time, event in all_times:
                count = 0.0
                total = 0.0
                for pattime, patevent in times[i]:
                    if pattime >= time:
                        count += 1
                        total += 1
                    elif patevent:
                        total += 1
                if not len(times[i]):
                    alive[i].append(0)
                else:
                    alive[i].append(count / total) # Probability
                #alive[i].append(count) # Actual counts
                if int(time) > prev:
                    prev = int(time)
                    ticklabels[prev] += '\n{0}'.format(int(count))
                    #ticklabels[prev] = ticklabels[prev].strip() #Remove possible leading new line on first row

        #Now plot times vs alive
        fig = plt.figure()
        fig.subplots_adjust(bottom = 0.22) #Move the plot up a bit so x-axis labels fit
        ax = fig.add_subplot(111)
        ps = []
        labels = []
        styles = ['r-', 'g-', 'b-', 'k-']
        #Make sure styles is just as long as alive
        while len(styles) < len(alive):
            styles.append('k-')

        for i in reversed(xrange(len(alive))): #Do best chance first, so they appear highest in legend
            ps.append(ax.plot([x[0] for x in all_times], alive[i], styles[i]))
            labels.append(str(alive[i][-1])[:4])

        #leg = ax.legend(ps, labels, 'lower left')
        ax.set_xlabel("Time, years")
        ax.set_ylabel("Survival ratio")
        ax.set_title("Kaplan-Meier survival curve\nThresholds: " + str([str(t)[:4] for t in sorted(threshold, reverse = True)]))

        #Add a few values to the right side of the plot
        final_ticks = []
        lower = 1.0
        for i in reversed(xrange(len(alive))):
            final_ticks.append(alive[i][-1])
            if alive[i][-1] < lower:
                lower = alive[i][-1]
        ax_right = plt.twinx(ax)
        ax_right.set_yticks(final_ticks)
        ax_right.set_yticklabels(labels)

        #Set limits on both
        ax.set_ylim(ymin = lower, ymax = 1.0)
        ax_right.set_ylim(ymin = lower, ymax = 1.0)

        #Add patient counts
        ax.set_xticklabels(ticklabels)

        if show_plot:
            show()

        return threshold

def scatter(data_x, data_y, events = None, show_plot = True, gridsize = 50, mincnt = 0, x_label = '', y_label = ''):
    '''
    It is assumed that the x-axis contains the target data, and y-axis the computed outputs.
    If events is not None, then any censored data (points with a zero in events) will not be able to go above the diagonal.
    Reason for that being that a censored data is "correct" if the output for that data is greater or equal to the diagonal point.
    Gridsize determines how many hexagonal bins are used (on the x-axis. Y-axis is determined automatically to match)
    mincnt is the minimum number of hits a bin needs to be plotted.
    '''
    if not len(data_x) == len(data_y) or (events is not None and not len(data_x) == len(events)):
        raise ValueError('Lengths of arrays do not match!')

    xmin = data_x.min()
    xmax = data_x.max()
    ymin = data_y.min()
    ymax = data_y.max()

    if events is not None:
        x_range = xmax - xmin
        y_range = ymax - ymin
        slope = y_range / x_range
        for event, i in zip(events, xrange(len(data_y))):
            diagonal_point = ymin + slope * (data_x[i] - xmin)
            if event == 0 and data_y[i] > diagonal_point:
                #print('Shifting {0},{1} to {2}'.format(data_x[i], data_y[i], diagonal_point))
                data_y[i] = diagonal_point

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pc = ax.hexbin(data_x, data_y, bins = 'log', cmap = cm.jet, gridsize = gridsize, mincnt = mincnt)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title("Scatter plot heatmap, taking censored into account") if events is not None else ax.set_title("Scatter plot heatmap")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cb = fig.colorbar(pc, ax = ax)
    cb.set_label('log10(N)')
    #ax.plot([xmin, xmax], [ymin, ymax], 'r-') #Print slope

    if show_plot:
        show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        #filename = "/home/gibson/jonask/Projects/Kaplan-Meier/genetic.csv"
        filename = "/home/gibson/jonask/Projects/Kaplan-Meier/censored_3node.csv"
    else:
        filename = sys.argv[1]

    data = np.array(read_data_file(filename, ","))
    D, t = parse_data(data, inputcols = (2, 3, 4, 5, 6, 7, 8, 9, 10), ignorerows = [0], normalize = False)

    kaplanmeier(D, 2, 3, -1, show_plot = False)


