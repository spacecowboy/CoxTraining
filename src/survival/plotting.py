from kalderstam.util.filehandling import read_data_file, parse_data
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional
import numpy as np
from random import random

def kaplanmeier(data, time_column, event_column, output_column, threshold = None):
    ''' Idea is to plot number of patients still alive on the y-axis,
    against the time on the x-axis. The time column specifies which
    axis is the time. Event column should be a binary value, indicating
    1 for an event and 0 for (right)censored. 
    Output column specifies the network output column
    and threshold the value(s) to divide the groups by (network output).
    If no threshold is specified, the set will be divided into 2 equally large
    sets, divided by the median.
    '''

    if plt:
        #Divide set
        times = [[] for _ in xrange(2)]
        alive = [[] for _ in xrange(2)]

        if threshold is None:
            threshold = np.median(data[:, output_column])

        for time, event, value in data[:, (time_column, event_column, output_column)]:
            if value < threshold:
                times[0].append((time, event))
            else:
                times[1].append((time, event))

        #times[0] = sorted(times[0][:, 0])
        times[0] = sorted(times[0], key = lambda x: x[0])
        #times[1] = sorted(times[1][:, 0])
        times[1] = sorted(times[1], key = lambda x: x[0])
        #Now make list of all time indices, this is just a convenience for plotting points
        all_times = sorted(times[0] + times[1], key = lambda x: x[0])

        #Count how many are alive at each time
        for i in xrange(2):
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

        #Now plot times vs alive
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ps = []
        labels = []
        styles = ['b-', 'r-']
        for i in xrange(2):
            ps.append(ax.plot([x[0] for x in all_times], alive[i], styles[i]))
            labels.append(str(alive[i][-1]))
        leg = ax.legend(ps, labels, 'upper right')
        #leg = ax.legend('upper right')
        ax.set_xlabel("Time, years")
        ax.set_ylabel("Survival ratio")
        ax.set_title("Kaplan-Meier survival curve, threshold: " + str(threshold))
        plt.show()

def divide_and_plot(data, target_column, num = 2):
    ''' Idea is to plot number of patients still alive on the y-axis,
    against the time on the x-axis. The target column specifies which
    axis is the time. The data set is divided into num- parts and
    plotted against each other.
    '''

    if plt:
        #Divide set
        times = [[] for _ in xrange(num)]
        alive = [[] for _ in xrange(num)]

        avg_time = np.average(data[:, target_column]) - 2

        for time in data[:, target_column]:
            #if time < avg_time:
            if time < 5.0 * random():
                times[0].append(time)
            else:
                times[1].append(time)

        times[0] = sorted(times[0])
        times[1] = sorted(times[1])
        #Now make list of all time indices
        all_times = sorted(times[0] + times[1])

        #Count how many are alive at each time
        for i in xrange(num):
            for time in all_times:
                count = 0.0
                for pattime in times[i]:
                    if pattime >= time:
                        count += 1
                alive[i].append(count / len(times[i]))

        #Now plot times vs alive
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in xrange(num):
            ax.plot(all_times, alive[i], 'b-')
        #plt.legend([p], [str(alive[1, -1])], loc = "upper right")
        leg = ax.legend((str(alive[0, -1]), str(alive[1, -1])), 'upper right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival ratio")
        ax.set_title("Survival plot")
        plt.show()

if __name__ == '__main__':
    filename = "/home/gibson/jonask/Projects/Kaplan-Meier/genetic.csv"
    #P, T = parse_file(filename, targetcols = [4], inputcols = [-1, -2, -3, -4], ignorerows = [0], normalize = True)

    data = np.array(read_data_file(filename, ", "))
    D, t = parse_data(data, inputcols = (2, 3, 4, 5, 6, 7, 8, 9, 10), ignorerows = [0], normalize = False)
    #data = np.array(data[1:])

    #divide_and_plot(T, 0)

    #kaplanmeier(D, 2, 3, -1, 4.0)
    kaplanmeier(D, 2, 3, -1)
