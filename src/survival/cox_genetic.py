from survival.cox_error import get_C_index
from kalderstam.util.decorators import benchmark_adv

@benchmark_adv
def c_index_error(target, result):
    '''Used in genetic training.'''
    #len(target) first to compensate for internals in genetic training
    #abs( - 0.5) to make both "positive" and "negative" C_index work, since they do
    C = get_C_index(target, result)
    if C < 0.51:
        #dont want these right now, return 100 as error
        #also avoids division by zero below
        return 100.0

    return len(target) / abs(C - 0.5) - 2 * len(target) #return inverse, error should be low if c_index is high. last minus term makes the minimum zero and not two.
