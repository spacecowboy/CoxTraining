'''
Created on Aug 23, 2011

@author: jonask
'''
import unittest
import numpy as np
from survival.cox_error_in_c import get_C_index
from survival.cox_genetic import c_index_error

class Test(unittest.TestCase):

    def generateRandomTestData(self, number):
        outputs = np.random.random((number, 2))
        for i in xrange(len(outputs)):
            outputs[i, 1] = np.random.randint(0, 2) #inclusive, exclusive

        return outputs

    def testGeneticCindexError(self):
        T = self.generateRandomTestData(1000)
        outputs = self.generateRandomTestData(1000)
        c_index = get_C_index(T, outputs)
        rand_error = c_index_error(T, outputs) / len(T)
        test_error = 1 / abs(c_index - 0.5) - 2
        print("rand_error = ", rand_error, "test value = ", test_error, "c_index = ", c_index)
        assert(rand_error == test_error)

        T[:, 0] = np.arange(len(T))
        outputs = T
        rev_outputs = outputs[::-1]

        c_index = get_C_index(T, outputs)
        ord_error = c_index_error(T, outputs) / len(T)
        test_error = 1 / abs(c_index - 0.5) - 2
        print("ordered_error = ", ord_error, "test value = ", test_error, "c_index = ", c_index)
        assert(ord_error == test_error)

        c_index = get_C_index(T, rev_outputs)
        rev_error = c_index_error(T, rev_outputs) / len(T)
        test_error = 1 / abs(c_index - 0.5) - 2
        print("reversed_error = ", rev_error, "test value = ", test_error, "c_index = ", c_index)
        assert(rev_error == test_error)

        assert(ord_error == rev_error)

        T[:, 0] = np.arange(len(T))
        T[0, 1], T[-1, 1] = 1, 1 #Make sure they are non-censored
        outputs = T.copy()
        outputs[0], outputs[-1] = outputs[-1], outputs[0]
        rev_outputs = outputs[::-1]

        c_index = get_C_index(T, outputs)
        ord_error = c_index_error(T, outputs) / len(T)
        test_error = 1 / abs(c_index - 0.5) - 2
        print("1_off_error = ", ord_error, "test value = ", test_error, "c_index = ", c_index)
        assert(ord_error == test_error)

        assert(ord_error > 0)

        c_index = get_C_index(T, rev_outputs)
        rev_error = c_index_error(T, rev_outputs) / len(T)
        test_error = 1 / abs(c_index - 0.5) - 2
        print("1_off_reversed_error = ", rev_error, "test value = ", test_error, "c_index = ", c_index)
        assert(rev_error == test_error)

        assert(rev_error > 0)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
