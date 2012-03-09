from __future__ import division
from kalderstam.util.filehandling import read_data_file, parse_data
try:
    import matplotlib
    matplotlib.use('GTKAgg')
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

def kaplanmeier(data = None, time_column = None, event_column = None, output_column = None, time_array = None,
                event_array = None, output_array = None, threshold = None, even = False, show_plot = True):
    ''' Idea is to plot number of patients still alive on the y-axis,
    against the time on the x-axis. The time column specifies which
    axis is the time. Event column should be a binary value, indicating
    1 for an event and 0 for (right)censored. 
    Output column specifies the network output column
    and threshold the value(s) to divide the groups by (network output).
    If no threshold is specified, the set will be divided into 2 equally large
    sets, divided by the median.
    '''
    NUMBER_OF_TICKS = 6    
    
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

        ticklabels = ["{0}y".format(int(year[0])) for year in all_times]
        bestticklabels = ["{0}y".format(int(year[0])) for year in all_times]

        #Count how many are alive at each time
        for i in sorted(xrange(len(times)), reverse = True): #Take best survivors first
            prev = -99999
            for all_times_index, (time, event) in enumerate(all_times):
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
                ticklabels[all_times_index] += '\n{0}'.format(int(count))
                #if int(time) > prev:
                #    prev = int(time)
                #    ticklabels[prev] += '\n{0}'.format(int(count))
                    #ticklabels[prev] = ticklabels[prev].strip() #Remove possible leading new line on first row
                    
        #Find at which time we have the greatest difference, if we have only two groups
        best_limit = 0
        best_difference = 0
        if len(alive) == 2:
            limit = 0 # Yes, because we want it +1 at the end anyway fornon-inclusive slices
            for chance_a, chance_b in zip(alive[0], alive[1]):
                limit += 1
                diff = abs(chance_a - chance_b)
                if diff > best_difference:
                    best_difference = diff
                    best_limit = limit

        #Now plot times vs alive
        fig = plt.figure()
        fig.subplots_adjust(bottom = 0.22) #Move the plot up a bit so x-axis labels fit
        if len(alive) == 2:
            bestax = fig.add_subplot(211)
            bestax_right = plt.twinx(bestax)
        ax = fig.add_subplot(212)
        ax_right = plt.twinx(ax)
        ps = []
        labels = []
        best_labels = []
        styles = ['r-', 'g-', 'b-', 'k-']
        #Make sure styles is just as long as alive
        while len(styles) < len(alive):
            styles.append('k-')

        for i in reversed(xrange(len(alive))): #Do best chance first, so they appear highest in legend
            ps.append(ax.plot([x[0] for x in all_times], alive[i], styles[i]))
            best_labels.append(str(alive[i][best_limit-1])[:4])
            bestax.plot([x[0] for x in all_times[:best_limit]], alive[i][:best_limit], styles[i])

        #leg = ax.legend(ps, labels, 'lower left')
        ax.grid(True, 'both')
        ax.set_xlabel("Time, years")
        ax.set_ylabel("Survival ratio")
        if (len(alive) == 2):
            bestax.set_title("Kaplan-Meier survival curve\nThresholds: " + str([str(t)[:4] for t in sorted(threshold,
                                                                                reverse = True)]))
        else:
            ax.set_title("Kaplan-Meier survival curve\nThresholds: " + str([str(t)[:4] for t in sorted(threshold,
                                                                            reverse = True)]))
                                                                        
        if len(alive) == 2:
            bestax.grid(False, 'both')
            #bestax.set_xlabel("Time, years")
            bestax.set_ylabel("Largest difference")   
            
        #Set limits
        
        if len(alive) == 2:
            print "xmin = " + str(all_times[0][0]) + " "  + str(all_times[1][0])
            bestax.set_xlim(xmin = all_times[0][0], xmax = [x[0] for x in all_times][best_limit-1])
            bestax.minorticks_off()
            bestax.set_yticklabels([])
        
        #Se the ticklabels
        print(len(all_times))
        ticks = []
        final_tick_index = 0
        major_ticks = []
        if len(ticklabels) == NUMBER_OF_TICKS:
            major_ticks = ticklabels
        else:
            every_nth = best_limit / (NUMBER_OF_TICKS -1)
            print("Nth: " + str(every_nth) + " , " + str(int(every_nth)))
            every_nth = int(every_nth)
            for label, tick_index in zip(ticklabels, xrange(len(ticklabels))):
                #major_num includes the first zero, so minus one
                if tick_index == 0 or tick_index % every_nth == 0:
                    major_ticks.append(label)
                    ticks.append(all_times[tick_index][0])
                    final_tick_index = tick_index
                    print("index: " + str(tick_index) + " label: " + label)
          
        
        ax.set_xlim(xmin = min(ticks), xmax = max(ticks))
        ax.set_xticks(ticks)
        ax.set_xticklabels(major_ticks)
        
        
        if len(alive) == 2:
            bmajor_ticks = []
            ticks = []
           
            bevery_nth = best_limit / (NUMBER_OF_TICKS -1)
            print("Nth: " + str(bevery_nth) + " , " + str(int(bevery_nth)))
            for label, tick_index in zip(bestticklabels[:best_limit], xrange(best_limit-1)):
                #NUMBER_OF_TICKS includes the first zero, so minus one
                if tick_index == 0 or tick_index % int(bevery_nth) == 0:
                    bmajor_ticks.append(label)
                    ticks.append(all_times[tick_index][0])
                    print("index: " + str(tick_index) + " label: " + str(label))
                    
            bestax.set_xticks(ticks)
            bestax.set_xticklabels(bmajor_ticks)
            
        #Add a few values to the right side of the plot
        best_final_ticks = []
        final_ticks = []
        lower = 1.0
        best_lower = 1.0
        best_upper = 0.0
        for i in reversed(xrange(len(alive))):
            labels.append(str(alive[i][final_tick_index])[:4])
            final_ticks.append(alive[i][final_tick_index])
            if alive[i][final_tick_index] < lower:
                lower = alive[i][final_tick_index]
            best_final_ticks.append(alive[i][best_limit-1])
            if alive[i][best_limit-1] < best_lower:
                best_lower = alive[i][best_limit-1]
            if alive[i][best_limit-1] > best_upper:
                best_upper = alive[i][best_limit-1]
            
        
        ax_right.set_yticks(final_ticks)
        ax_right.set_yticklabels(labels)
        
        if len(alive) == 2:
            bestax.set_ylim(ymin = best_lower, ymax = 1.0)
            bestax_right.set_ylim(ymin = best_lower, ymax = 1.0)
            bestax_right.set_yticks(best_final_ticks)
            bestax_right.set_yticklabels(best_labels)
            
        #Set limits on both
        ax.set_ylim(ymin = lower, ymax = 1.0)
        ax_right.set_ylim(ymin = lower, ymax = 1.0)
        ax.minorticks_off()

        if show_plot:
            show()

        return threshold
        
def calc_line(x, y):
    '''
    y = mx + c
    We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]. Now use lstsq to solve for p:
    Returns m, c
    '''
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y)[0]
    
def scatter(data_x, data_y, events = None, show_plot = True, gridsize = 30, mincnt = 0, x_label = '', y_label = '', title= '', ax = None, plotSlope = None):
    '''
    It is assumed that the x-axis contains the target data, and y-axis the computed outputs.
    If events is not None, then any censored data (points with a zero in events) will not be able to go above the diagonal.
    Reason for that being that a censored data is "correct" if the output for that data is greater or equal to the diagonal point.
    The diagonal is calculated from the non-censored data (all if no events specified) using least squares linear regression.
    Gridsize determines how many hexagonal bins are used (on the x-axis. Y-axis is determined automatically to match)
    mincnt is the minimum number of hits a bin needs to be plotted.
    '''    
    if not len(data_x) == len(data_y) or (events is not None and not len(data_x) == len(events)):
        raise ValueError('Lengths of arrays do not match!')

    xmin = data_x.min()
    xmax = data_x.max()
    ymin = data_y.min()
    ymax = data_y.max()
    
    if plotSlope is None:
        #Nothing has been specified, we do it if we have events
        plotSlope = True if events is not None else False
    
    #For plotting reasons, we need to have these sorted
    if events is None:
        sorted_x_y = [[data_x[i], data_y[i]] for i in xrange(len(data_x))]
    else:
        sorted_x_y = [[data_x[i], data_y[i], events[i]] for i in xrange(len(data_x))]
    sorted_x_y.sort(lambda x, y: cmp(x[0], y[0])) #Sort on target data
    sorted_x_y = np.array(sorted_x_y)
    #Calculate the regression line (if events is None weneed it later)
    slope, cut = calc_line(sorted_x_y[:, 0], sorted_x_y[:, 1])

    if events is not None:
        #We must calculate the diagonal from the non-censored
        non_censored_x = sorted_x_y[:, 0][sorted_x_y[:, 2] == 1]
        non_censored_y = sorted_x_y[:, 1][sorted_x_y[:, 2] == 1]
        
        ymin = non_censored_y.min()
        ymax = non_censored_y.max()
        
        slope, cut = calc_line(non_censored_x, non_censored_y)
        
        #And then no censored point can climb above the diagonal. Their value is the percentage of their comparisons
        #in the C-index which are successful
        for i in xrange(len(sorted_x_y)):
            target, output, event = sorted_x_y[i]
            if event == 0:
                #Compare with all previous non-censored and calculate ratio of correct comparisons
                total = num_of_correct = 0
                for prev_target, prev_output, prev_event in sorted_x_y[:i]:
                    if prev_event == 1:
                        total += 1
                        if prev_output <= output: #cmp(prev_output, output) < 1
                            num_of_correct += 1
                            
                #Now we have the ratio
                #If no previous non-censored exist (possible for test sets for example) we move to zero.
                total = 1.0 if total < 1 else total
                ratio = num_of_correct / total
                
                #Move the point
                diagonal_point = cut + slope * target
                sorted_x_y[i][1] = ymin + ratio * (diagonal_point - ymin)

    axWasNone = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        axWasNone = True
    pc = ax.hexbin(sorted_x_y[:, 0], sorted_x_y[:, 1], bins = 'log', cmap = cm.jet,
                   gridsize = gridsize, mincnt = mincnt)
    ax.axis([xmin, xmax, ymin, ymax])
    #line_eq = "Line: {m:.3f} * x + {c:.3f}".format(m=slope, c=cut)
    #ax.set_title("Scatter plot heatmap, taking censored into account\n" + line_eq) if events is not None else \
    #    ax.set_title("Scatter plot heatmap\n" + line_eq)
    if axWasNone:
        cb = fig.colorbar(pc, ax = ax)
        cb.set_label('log10(N)')
    if plotSlope:
        ax.plot(sorted_x_y[:, 0], slope*sorted_x_y[:, 0] + cut, 'r-') #Print slope
    #ax.scatter(sorted_x_y[:, 0], sorted_x_y[:, 1], c='g')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if axWasNone and show_plot:
        show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        #filename = "/home/gibson/jonask/Projects/Kaplan-Meier/genetic.csv"
        #filename = "/home/gibson/jonask/Projects/Kaplan-Meier/censored_3node.csv"
        #filename = "/home/gibson/jonask/Projects/Experiments/src/cox_com_3tanh_output"
        filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data/ann/" + \
                    ".test_.15_tanh_1323233958_Two_thirds_of_the_n4369_dataset_with_logs_lymf_5YEAR.cvs"
    else:
        filename = sys.argv[1]

    data = np.array(read_data_file(filename, "\t"))
    D, t = parse_data(data, inputcols = (0, 1, 2), ignorerows = [0], normalize = False)

    kaplanmeier(D, 0, 2, 1, show_plot = False)
    scatter(D[:, 0], D[:, 1], D[:, 2], x_label = 'Target Data', y_label = 'Model Correlation', show_plot = False)
    scatter(D[:, 0], D[:, 1], x_label = 'Target Data', y_label = 'Model Output')
    
