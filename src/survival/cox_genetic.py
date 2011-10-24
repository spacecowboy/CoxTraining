from survival.cox_error_in_c import get_C_index, get_weighted_C_index

def c_index_error(target, result):
    '''Used in genetic training.
    multiplied by length of target array because it is divided by the length of the target array in the genetic algorithm.'''
    #len(target) first to compensate for internals in genetic training
    #abs( - 0.5) to make both "positive" and "negative" C_index work, since they do
    C = get_C_index(target, result)

    return __inversed__(C, len(target))

def weighted_c_index_error(target, result):
    '''Used in genetic training.
    multiplied by length of target array because it is divided by the length of the target array in the genetic algorithm.'''
    #len(target) first to compensate for internals in genetic training
    #abs( - 0.5) to make both "positive" and "negative" C_index work, since they do
    C = get_weighted_C_index(target, result)

    return __inversed__(C, len(target))

def __inversed__(perf, length):
    '''Used in genetic training.
    multiplied by length of target array because it is divided by the length of the target array in the genetic algorithm.'''
    #len(target) first to compensate for internals in genetic training
    #abs( - 0.5) to make both "positive" and "negative" C_index work, since they do
    try:
        #A C of 0 is achievable if the network evolves to always output the same value regardless of input. Thus we can't
        #trust values lower than 0.5.
        retval = 1.0 / perf
    except (ZeroDivisionError):
        retval = 9000.0 #Very unlikely, but still possible

    return length * retval
