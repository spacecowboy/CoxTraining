'''
Created on Nov 3, 2011

@author: jonask

Based on the paper of Lian Yan, David Verbel and Olivier Saidi: Predicting Prostate Cancer Recurrence via Maximizing the concordance Index
'''
from __future__ import division

def block_func(inputs, targets, block_size, outputs, block_members, valid_pairs, weighted_sum):
    '''If not equal to all, must recompute valid_pairs and weighted sum'''
    if block_size == 0 or block_size == len(targets):
        return {'valid_pairs': valid_pairs, 'weighted_sum': weighted_sum}
    else:
        return pre_loop_func(None, None, targets[block_members], block_size)

def epoch_func(net, test_inputs, test_targets, block_size, epoch, valid_pairs, weighted_sum):
    '''Only passes the pre_loop variables on so they get into total_error'''
    return {'valid_pairs': valid_pairs, 'weighted_sum': weighted_sum}

def pre_loop_func(net, inputs, targets, block_size):
    '''This function determines the Omega collection. Which pairs should be compared.
    Returns a list of tuples where each tuple is (i, j). E.g. i will be non-censored and have lower recurrence time than j. J might be censored, or not.'''
    valid_pairs = []
    for i in xrange(len(targets)):
        for j in xrange(len(targets)):
            if i != j:
                if targets[i, 1] == 1 and targets[i, 0] < targets[j, 0]:
                    valid_pairs.append((i, j))

    D = sum_of_weighted_comparisons(targets, valid_pairs)

    return {'valid_pairs': valid_pairs, 'weighted_sum' : D}

def weighted_comparison(target_i, target_j):
    '''Simply to minimize errors. -(Ti - Tj)'''
    return (-target_i + target_j)

def diff_weighted_comparison(i, diff_index):
    if i == diff_index:
        return - 1
    else:
        return 0

def sum_of_weighted_comparisons(targets, valid_pairs):
    '''Known as D, equation (8), in the paper.
    The sum of all weights in Omega, the collection determining which pairs can be compared.'''
    D = 0
    for i, j in valid_pairs:
        D += weighted_comparison(targets[i, 0], targets[j, 0])
    return D

def diff_sum_of_weighted_comparisons(valid_pairs, diff_index):
    D_diff = 0
    for i, j in valid_pairs:
        D_diff += diff_weighted_comparison(i, diff_index)
    return D_diff

def soft_comparison(output_i, output_j, threshold = 0.2, rate = 2):
    '''Known as R, equation (5), in the paper. Returns an approximation to the weighted comparison.
    threshold = gamma
    rate = n'''
    if output_i - output_j < threshold:
        return (weighted_comparison(output_i, output_j) + threshold) ** rate
    else:
        return 0

def diff_soft_comparison(outputs, i, j, diff_index, threshold = 0.2, rate = 2):
    if outputs[i, 0] - outputs[j, 0] < threshold:
        return (diff_weighted_comparison(i, diff_index)) * rate * (weighted_comparison(outputs[i, 0], outputs[j, 0]) + threshold) ** (rate - 1)
    else:
        return 0

def total_error(targets, outputs, valid_pairs = None, weighted_sum = None):
    '''Equation (7) in the paper. Total index. 1 is perfect, 0.5 is random, 0 is just crap.'''
    if valid_pairs is None:
        valid_pairs = pre_loop_func(None, None, targets, None)['valid_pairs']

    if weighted_sum is None:
        weighted_sum = sum_of_weighted_comparisons(targets, valid_pairs)

    C_weighted = 0
    for i, j in valid_pairs:
        C_weighted += weighted_comparison(targets[i, 0], targets[j, 0]) * soft_comparison(outputs[i, 0], outputs[j, 0])

    C_weighted /= weighted_sum

    return C_weighted

def derivative(targets, outputs, diff_index, valid_pairs, weighted_sum):
    #First the denominator
    dCw = total_error(targets, outputs, valid_pairs, weighted_sum) / weighted_sum * diff_sum_of_weighted_comparisons(valid_pairs, diff_index) * -1 #Minus sign because it came from the denominator
    #Now the top side
    top = 0
    for i, j in valid_pairs:
        top += diff_weighted_comparison(i, diff_index) * soft_comparison(outputs[i, 0], outputs[j, 0]) + weighted_comparison(targets[i, 0], targets[j, 0]) * diff_soft_comparison(outputs, i, j, diff_index)
    dCw += top / weighted_sum

    return dCw
