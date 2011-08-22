from survival.cox_error_in_c import get_C_index

def c_index_error(target, result):
    '''Used in genetic training.
    multiplied by length of target array because it is divided by the length of the target array in the genetic algorithm.'''
    #len(target) first to compensate for internals in genetic training
    #abs( - 0.5) to make both "positive" and "negative" C_index work, since they do
    C = get_C_index(target, result)
    if C < 0.51:
        #dont want these right now, return 100 as error
        #also avoids division by zero below
        retval = 100.0
    else:
        retval = (1 / abs(C - 0.5) - 2.0) #return inverse, error should be low if c_index is high. last minus term makes the minimum zero and not two.

    return len(result) * retval
