'''
Created on Sep 1, 2011

@author: jonask
'''

#Defines the node, and the network
import numpy
from kalderstam.neural.fast_network import Node as node
from kalderstam.neural.network import committee as normal_com, \
    connect_nodes, network as normal_net

def build_feedforward_committee(size = 8, input_number = 2, hidden_number = 2, output_number = 1, hidden_function = "tanh", output_function = "logsig"):
    net_list = [build_feedforward(input_number, hidden_number, output_number, hidden_function, output_function) for n in xrange(size)]
    return committee(net_list)

def build_feedforward_multilayered(input_number = 2, hidden_numbers = [2], output_number = 1, hidden_function = "tanh", output_function = "logsig"):
    net = network()
    net.num_of_inputs = input_number
    inputs = range(input_number)

    #Hidden layers
    prev_layer = inputs
    for hidden_number in hidden_numbers:
        current_layer = []
        for i in xrange(int(hidden_number)):
            hidden = node(hidden_function)
            connect_nodes(hidden, prev_layer)
            net.hidden_nodes.append(hidden)
            current_layer.append(hidden)
        prev_layer = current_layer

    #Output nodes
    for i in xrange(int(output_number)):
        output = node(output_function)
        connect_nodes(output, prev_layer)
        net.output_nodes.append(output)

    return net

def build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = "tanh", output_function = "logsig"):
    net = network()
    net.num_of_inputs = input_number
    inputs = range(input_number)

    #Hidden layer
    for i in xrange(int(hidden_number)):
        hidden = node(hidden_function)
        connect_nodes(hidden, inputs)
        net.hidden_nodes.append(hidden)

    #Output nodes
    for i in xrange(int(output_number)):
        output = node(output_function)
        connect_nodes(output, net.hidden_nodes)
        net.output_nodes.append(output)

    return net

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

    def risk_eval(self, input):
        '''
        Returns the average index value for input over all members of the committee.
        '''
        avg_index = 0.0
        for net in self.nets:
            output = net.update(input)
            index = len(net.trn_set[net.trn_set < output]) # Length of the array of all values less than output = index where output would be placed
            #Normalize it
            avg_index += index / float((len(net.trn_set) + 1)) # +1 to make sure the maximum is 1.0 if the input is placed last

        return avg_index / float(len(self))

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
