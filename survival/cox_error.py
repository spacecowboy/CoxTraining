from numpy import log, exp
import logging
import numpy as np
#import kalderstam.util.graphlogger as glogger
from cox_error_in_c import derivative_beta as cderivative_beta, get_slope as cget_slope, get_C_index as get_cC_index
import kalderstam.util.graphlogger as glogger
from kalderstam.util.numpyhelp import indexOf
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional
except RuntimeError:
    plt = None

logger = logging.getLogger('kalderstam.neural.error_functions')

shift = 4 #Also known as Delta, it's the handwaving variable.

def get_C_index(T, outputs):
    """
    DO NOT USE THIS VERSION. Instead use the much faster version in the c-module.
    
    Compare that the outputs are sorted 'correctly'.
    if T[x,0] < T[y,0] and X[x] < X[y] or T[x,0] > T[y,0] and X[x] > X[y], plus 1. Finally divide by the number of comparisons made.
    Non-censored points can be compared with all other non-censored points and all later censored points.
    Censored points can only be compared to earlier non-censored points."""
    total = 0
    sum = 0
    for x in xrange(len(T)):
        for y in xrange(len(T)):
            if x == y:
                continue #Don't compare with itself
            if T[x, 1] == 1 and (T[y, 1] == 1): #Non-censored, compare with all other non-censored
                if T[x, 0] < T[y, 0]: #all non-censored will be compared eventually, but only once
                    total += 1
                    if outputs[x, 0] < outputs[y, 0]:
                        sum += 1
            elif T[x, 1] == 1 and (T[y, 1] == 0) and T[x, 0] < T[y, 0]: #Non-censored, compare with later censored
                total += 1
                if outputs[x, 0] <= outputs[y, 0]:
                    sum += 1

    sum /= float(total)
    return sum


def generate_timeslots(T):
    """Setting for_censored to true will reverse the behaviour and the function generates timeslots for censored data instead of for non-censored.
    Used for plotting the order-scatter"""
    timeslots = np.array([], dtype = np.int64)
    for x_index in xrange(len(T)):
        event = T[x_index][1]
        if event == 1: #Else it was censored
            time = T[x_index][0]
            if len(timeslots) == 0:
                timeslots = np.insert(timeslots, 0, x_index)
            else:
                added = False
                #Find slot
                for index in xrange(len(timeslots)):
                    time_index = timeslots[index]
                    if time < T[time_index, 0]:
                        timeslots = np.insert(timeslots, index, x_index)
                        added = True
                        break
                if not added:
                    #Reached the end, insert here
                    timeslots = np.append(timeslots, x_index)

    return timeslots

def get_risk_groups(T, timeslots):
    """T is the target vector of [ [time_to_event/censoring, event] ],
     where event is 1 if there was an event, 0 if it was censored.
     A risk group is defined as those who disappeared from the study up to time t."""
    risk_groups = []
    # Sort T on the time?
    for i in timeslots:
        group = np.array([], dtype = np.int64)
        #Iterate over T and add all with a time less than T[i]
        for j in xrange(len(T)):
            if T[j][0] <= T[i][0]:
                group = np.append(group, j)
        risk_groups.append(group)
    return risk_groups

def generate_random_data(number):
    outputs = np.random.random((number, 2))
    for i in xrange(len(outputs)):
        outputs[i, 1] = np.random.randint(0, 2) #inclusive, exclusive
    timeslots = generate_timeslots(outputs)

    return (outputs, timeslots)

def censor_rndtest(T, ratio):
    temp = [[1.5, 0] for _ in T]
    Tc = np.array(temp, dtype = np.float64)
    for i in xrange(len(T)):
        if np.random.rand() <= ratio:
            e = 0
            Tc[i] = (np.random.rand() * T[i, 0], e)#also cut time
        else:
            e = 1
            Tc[i] = (T[i, 0], e)
    return Tc

def orderscatter(outputs, T, filename = "", marker = '+'):
    if plt:
        c_index = get_cC_index(T, outputs)

        T_copy = T.copy()
        T_copy[:, 0] = outputs[:, 0]

        plt.figure()
        plt.title('Scatter between the indices of the sorted target and sorted output arrays\n' + str(filename) + "\nC index = " + str(c_index) + "\nEvents in green, Censored in red.")
        plt.xlabel('Target index')
        plt.ylabel('Network index')

        largest = 0
        for x_index in xrange(len(T)):
            index_t = 0 #index in T
            index_o = 0 #index in outputs
            color = 'g'
            if not T[x_index, 1]:
                color = 'r'
            for cmp_index in xrange(len(T)):
                if T[x_index, 0] > T[cmp_index, 0]:
                    index_t += 1
                    if index_t > largest:
                        largest = index_t
                if T_copy[x_index, 0] > T_copy[cmp_index, 0]:
                    index_o += 1
            plt.plot(index_t, index_o, color + marker)

        plt.plot(range(largest), range(largest), 'r-')

def get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg):
    beta_force = 0
    for risk_group, z, w in zip(risk_groups, part_func, weighted_avg):
        beta_force += -1 / z * np.sum(np.exp(beta * outputs[risk_group, 0]) * outputs[risk_group, 0] ** 2) + w ** 2

    return beta_force

def get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups):
    output = outputs[output_index, 0]
    #print beta, output
    beta_out = exp(beta * output)
    y_force = 0
    for es, risk_group, z, w in zip(timeslots, risk_groups, part_func, weighted_avg):
        #glogger.debugPlot('Partition function', part_func[s], style = 'b+')
        #
        kronicker = 0
        if es == output_index:
            kronicker = 1
        if output_index in risk_group:
            dy_part = beta_out / z * (1 + beta * (output - w))
        else:
            dy_part = 0
        y_force += kronicker - dy_part
    return y_force

def derivative_error(beta, sigma):
    """dE/d(Beta*Sigma)"""
    exp_value = exp(shift - beta * sigma)
    de = -exp_value / (1 + exp_value)
    #glogger.debugPlot('Error derivative', de, style = 'r.')
    return de

def derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups):
    """Derivative of (Beta*Sigma) with respect to y(i)"""
    bs = derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups) * sigma + beta * derivative_sigma(sigma, output_index, outputs) #@UndefinedVariable
    #glogger.debugPlot('BetaSigma derivative', bs, style = 'g+')
    if np.isnan(bs) or np.isinf(bs):
        raise FloatingPointError('Derivative BetaSigma is Nan or Inf: ' + str(bs))
    return bs

def derivative_sigma(sigma, output_index, outputs):
    """Eq. 12, derivative of Sigma with respect to y(i)"""
    output = outputs[output_index, 0]
    ds = (output - outputs[:, 0].mean()) / (len(outputs) * sigma)
    #glogger.debugPlot('Sigma derivative', ds, style = 'b+')
    return ds

def derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups):
    """Eq. 14, derivative of Beta with respect to y(i)"""
    #glogger.debugPlot('Beta derivative', res, style = 'r+')
    return cderivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)

    #y_force = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)
    #beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

    #logger.info('OI:' + str(output_index) + ' B:' + str(beta / abs(beta)) + ' BF:' + str(beta_force / abs(beta_force)) + ' YF' + str(y_force / abs(y_force)))
    #return - y_force / beta_force

def get_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots):
    #cget_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots)
    result = 0
    for time_index in xrange(len(timeslots)):
        s = timeslots[time_index]
        output = outputs[s, 0]
        risk_outputs = outputs[risk_groups[time_index], 0]
        try:
            beta_risk[time_index] = np.exp(beta * risk_outputs)
        except FloatingPointError as e:
            logger.error("In get_slope for calc_beta: \n if beta is 40 and risk_output is -23, we will get an underflow.\n Setting numpy.seterr(under = 'warn') or 'ignore', will do set it to zero in that case.")
            raise(e)

        part_func[time_index] = np.sum(beta_risk[time_index])
        weighted_avg[time_index] = np.sum(beta_risk[time_index] * risk_outputs) / part_func[time_index]
        if np.isnan(weighted_avg[time_index]):
            #When beta is small enough, part_func will be zero. This means weighted avg is something divided by zero. raise exception
            raise FloatingPointError('Weighted avg (in get_slope) encountered a division by zero. Beta must be really small, Beta = ' + str(beta))
        result += (output - weighted_avg[time_index])

    return result

def calc_beta(outputs, timeslots, risk_groups):
    """Find the likelihood maximizing Beta numerically."""
    beta = 0.1 #Start with something small
    distance = 7.0 #Fairly large interval, we actually want to cross the zero

    beta_risk = [np.zeros(len(risk_groups[i])) for i in xrange(len(risk_groups))]
    part_func = np.zeros(len(timeslots))
    weighted_avg = np.zeros(len(timeslots))

    slope = cget_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots) #@UndefinedVariable

    not_started = True

    logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))
    #we will get overflow errors when beta goes above 710, but even 200 is completely unreasonable and means that beta will diverge. in that case, QUIT
    while abs(beta) < 200 and (abs(slope) > 0 or not_started) and abs(distance) > 0.0001: #Want positive beta, Some small limit close to zero, fix to make sure we try more than one value, stop when step size is too small
        not_started = False
        prev_slope = slope
        beta += distance
        slope = cget_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots) #@UndefinedVariable
        if slope * prev_slope < 0:
            #Different signs, we have passed the zero point, change directions and half the distance
            distance /= -2
        elif abs(slope) > abs(prev_slope):
            #If the new slope is bigger than the last, we are going in the wrong direction
            distance *= -1
        logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))

    if abs(beta) >= 200 or np.isnan(slope):
        raise FloatingPointError('Beta is diverging')
    logger.debug("Beta = " + str(beta))
    return beta, beta_risk, part_func, weighted_avg

def calc_sigma(outputs):
    """Standard deviation, just use numpy for it. need ALL results, from net.sim(inputs)"""
    sigma = outputs[:, 0].std()
    logger.debug("Sigma = " + str(sigma))
    return sigma

def total_error(target, result, error = 1, **kwargs):
    '''Intended for learning rate variance during training, must be precomputed from epoch-function.'''
    return error

def cox_error(beta, sigma):
    """E = ln(1 + exp(Delta - Beta*Sigma))."""
    return log(1 + exp(shift - beta * sigma))

def derivative(test_targets, outputs, index, beta, sigma, part_func, weighted_avg, beta_force, timeslots, risk_groups, **kwargs):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    return derivative_error(beta, sigma) * derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, index, outputs, timeslots, risk_groups)

def pre_loop_func(net, test_inputs, test_targets, block_size):
    np.seterr(all = 'raise') #I want errors!
    np.seterr(under = 'warn') #Except for underflows, just equate them to zero...

    if (block_size == 0 or block_size == len(test_targets)):
        timeslots = generate_timeslots(test_targets)
        risk_groups = get_risk_groups(test_targets, timeslots)
        return {'timeslots': timeslots, 'risk_groups': risk_groups}
    else:
        return {}

def epoch_func(net, test_inputs, test_targets, block_size, epoch, timeslots = None, risk_groups = None, **pre_loop_kwargs):
    outputs = net.sim(test_inputs)
    sigma = calc_sigma(outputs)
    if block_size != 0 and block_size != len(test_targets):
        timeslots = generate_timeslots(test_targets)
        risk_groups = get_risk_groups(test_targets, timeslots)
    beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

    error = cox_error(beta, sigma)

    c_index = get_cC_index(test_targets, outputs)
    logger.info("C index = " + str(c_index))

    glogger.debugPlot('Total error', error, style = 'b-')
    glogger.debugPlot('C index vs Epochs', c_index, style = 'b-')
    glogger.debugPlot('Sigma * Beta vs Epochs', beta * sigma, style = 'g-')
    #glogger.debugPlot('Sigma vs Epochs', sigma, style = 'b-')
    #glogger.debugPlot('Beta vs Epochs', beta, style = 'b-')
    logger.info('Beta*Sigma = ' + str(sigma * beta))
    return {'error': error}

def block_func(test_inputs, test_targets, block_size, outputs, block_members, timeslots = None, risk_groups = None, **kwargs):
    block_outputs = outputs[block_members]
    sigma = calc_sigma(block_outputs)
    if block_size != 0 and block_size != len(test_targets):
        timeslots = generate_timeslots(test_targets[block_members])
        risk_groups = get_risk_groups(test_targets[block_members], timeslots)
        retval = {'timeslots': timeslots, 'risk_groups': risk_groups}
    else:
        retval = {}
    beta, beta_risk, part_func, weighted_avg = calc_beta(block_outputs, timeslots, risk_groups)
    beta_force = get_beta_force(beta, block_outputs, risk_groups, part_func, weighted_avg)

    retval.update({'sigma':sigma, 'beta': beta, 'beta_risk': beta_risk, 'part_func': part_func, 'weighted_avg': weighted_avg, 'beta_force': beta_force})
    return retval
