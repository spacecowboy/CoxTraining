'''
Created on Sep 1, 2011

@author: jonask
'''

from __future__ import division
#Defines the node, and the network
import numpy
from kalderstam.neural.fast_network import Node as node
from kalderstam.neural.network import committee as normal_com, \
    connect_nodes, connect_node, network as normal_net

def build_feedforward_committee(size = 8, input_number = 2, hidden_number = 2, output_number = 1, hidden_function = "tanh", output_function = "logsig"):
    net_list = [build_feedforward(input_number, hidden_number, output_number, hidden_function, output_function) for n in xrange(size)]
    return committee(net_list)

def build_feedforward_multilayered(input_number = 2, hidden_numbers = [2], output_number = 1, hidden_function = "tanh", output_function = "logsig"):
    net = network()
    net.num_of_inputs = input_number
    inputs = range(input_number)
    
    biasnode = net.bias_node

    #Hidden layers
    prev_layer = inputs
    for hidden_number in hidden_numbers:
        current_layer = []
        for i in xrange(int(hidden_number)):
            hidden = node(hidden_function)
            connect_node(hidden, biasnode)
            connect_nodes(hidden, prev_layer)
            net.hidden_nodes.append(hidden)
            current_layer.append(hidden)
        prev_layer = current_layer

    #Output nodes
    for i in xrange(int(output_number)):
        output = node(output_function)
        connect_node(output, biasnode)
        connect_nodes(output, prev_layer)
        net.output_nodes.append(output)

    return net

def build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = "tanh", output_function = "logsig"):
    return build_feedforward_multilayered(input_number, [hidden_number],
            output_number, hidden_function, output_function)

class committee(normal_com):
    '''
    The committee can be built using normal networks. It will still be able to call these methods. Hence,
    using the committee is always safe. Using individual network methods can be risky.
    '''
    def set_training_sets(self, trn_input_sets):
        for net, inputs in zip(self.nets, trn_input_sets):
            results = net.sim(inputs)
            net.trn_set = results[:, 0] #To get rid of extra dimensions
            #Now sort the set
            net.trn_set = numpy.sort(net.trn_set)

    def risk_eval(self, input, net = None):
        '''
        Returns the average index value for input over all members of the committee, unless net is specified
        it will then perform a risk_eval for that network only (doesn't have to be a member even)
        '''
        if net is None:
            avg_index = 0.0
            for net in self.nets:
                output = net.update(input)
                index = len(net.trn_set[net.trn_set < output]) # Length of the array of all values less than output = index where output would be placed
                #Normalize it
                avg_index += index / float((len(net.trn_set) + 1)) # +1 to make sure the maximum is 1.0 if the input is placed last

            return avg_index / float(len(self))
        else:
            output = net.update(input)
            index = len(net.trn_set[net.trn_set < output]) # Length of the array of all values less than output = index where output would be placed
            #Normalize it
            return index / (len(net.trn_set) + 1)

class network(normal_net):
    '''
    Like a normal net, except it can also give a ranking based index of an input if
    its training_set has been set.
    '''

    def __init__(self):
        normal_net.__init__(self)
        self.trn_set = numpy.array([])

    def set_training_set(self, input_array):
        results = self.sim(input_array)
        self.trn_set = results[:, 0] #To get rid of extra dimensions
        #Now sort the set
        self.trn_set = numpy.sort(self.trn_set)

    def risk_eval(self, input):
        if len(self.trn_set) < 1:
            raise IndexError('No training set has been set!')
        else:
            output = self.update(input)
            index = len(self.trn_set[self.trn_set < output]) # Length of the array of all values less than output = index where output would be placed
            #Normalize it
            return index / float((len(self.trn_set) + 1)) # +1 to make sure the maximum is 1.0 if the input is placed last
            
def risk_eval(modded_net, test_input):
    if len(modded_net.trn_set) < 1:
        raise IndexError('No training set has been set!')
    else:
        output = modded_net.update(test_input)
        index = len(modded_net.trn_set[modded_net.trn_set < output]) # Length of the array of all values less than output = index where output would be placed
        #Normalize it
        return index / float((len(modded_net.trn_set) + 1)) # +1 to make sure the maximum is 1.0 if the input is placed last
