'''
Created on Aug 23, 2011

@author: jonask
'''
import unittest
import numpy as np
from survival.cox_error_in_c import get_C_index, get_weighted_C_index
from survival.cox_genetic import c_index_error, weighted_c_index_error

class Test(unittest.TestCase):

    def generateRandomTestData(self, number):
        outputs = np.random.random((number, 2))
        for i in xrange(len(outputs)):
            outputs[i, 1] = np.random.randint(0, 2) #inclusive, exclusive

        return outputs

    def testGeneticCindexError(self):
        print("\nC Error")
        T = self.generateRandomTestData(1000)
        outputs = self.generateRandomTestData(1000)
        c_index = get_C_index(T, outputs)
        rand_error = c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("rand_error = ", rand_error, "test value = ", test_error, "c_index = ", c_index)
        assert((rand_error - test_error) < 0.0001)

        T[:, 0] = np.arange(len(T))
        outputs = T
        rev_outputs = outputs[::-1]

        c_index = get_C_index(T, outputs)
        ord_error = c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("ordered_error = ", ord_error, "test value = ", test_error, "c_index = ", c_index)
        assert(ord_error == test_error)

        c_index = get_C_index(T, rev_outputs)
        rev_error = c_index_error(T, rev_outputs) / len(T)
        #test_error = 1 / c_index #Will give zero-division, set to 9000
        test_error = 9000.0
        print("reversed_error = ", rev_error, "test value = ", test_error, "c_index = ", c_index)
        assert(rev_error == test_error)

        assert(ord_error < rev_error)

        T[:, 0] = np.arange(len(T))
        T[0, 1], T[-1, 1] = 1, 1 #Make sure they are non-censored
        outputs = T.copy()
        outputs[0], outputs[-1] = outputs[-1], outputs[0]
        rev_outputs = outputs[::-1]

        c_index = get_C_index(T, outputs)
        ord_error = c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("1_off_error = ", ord_error, "test value = ", test_error, "c_index = ", c_index)
        assert(ord_error == test_error)

        assert(ord_error > 1)

        c_index = get_C_index(T, rev_outputs)
        rev_error = c_index_error(T, rev_outputs) / len(T)
        test_error = 1 / c_index
        print("1_off_reversed_error = ", rev_error, "test value = ", test_error, "c_index = ", c_index)
        assert(rev_error == test_error)

        assert(rev_error > 1)

    def testGeneticWeightedCindexError(self):
        print("\nWeighted C Error")
        T = self.generateRandomTestData(1000)
        outputs = self.generateRandomTestData(1000)
        c_index = get_weighted_C_index(T, outputs)
        rand_error = weighted_c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("rand_error = ", rand_error, "test value = ", test_error, "weighted_index = ", c_index)
        assert((rand_error - test_error) < 0.0001)

        T[:, 0] = np.arange(len(T))
        outputs = T
        rev_outputs = outputs[::-1]

        c_index = get_weighted_C_index(T, outputs)
        ord_error = weighted_c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("ordered_error = ", ord_error, "test value = ", test_error, "weighted_index = ", c_index)
        assert(ord_error == test_error)

        c_index = get_weighted_C_index(T, rev_outputs)
        rev_error = weighted_c_index_error(T, rev_outputs) / len(T)
        #test_error = 1 / c_index #Will give zero-division, set to 9000
        test_error = 9000.0
        print("reversed_error = ", rev_error, "test value = ", test_error, "weighted_index = ", c_index)
        assert(rev_error == test_error)

        assert(ord_error < rev_error)

        T[:, 0] = np.arange(len(T))
        T[0, 1], T[-1, 1] = 1, 1 #Make sure they are non-censored
        outputs = T.copy()
        outputs[0], outputs[-1] = outputs[-1], outputs[0]
        rev_outputs = outputs[::-1]

        c_index = get_weighted_C_index(T, outputs)
        ord_error = weighted_c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("1_off_ends_error = ", ord_error, "test value = ", test_error, "weighted_index = ", c_index)
        assert(ord_error == test_error)

        assert(ord_error > 1)
        ord_error_ends = ord_error

        c_index = get_weighted_C_index(T, rev_outputs)
        rev_error = weighted_c_index_error(T, rev_outputs) / len(T)
        test_error = 1 / c_index
        print("1_off_ends_reversed_error = ", rev_error, "test value = ", test_error, "weighted_index = ", c_index)
        assert(rev_error == test_error)

        assert(rev_error > 1)

        rev_error_ends = rev_error

        T[:, 0] = np.arange(len(T))
        T[len(T) / 2, 1], T[len(T) / 2 + 1, 1] = 1, 1 #Make sure they are non-censored
        outputs = T.copy()
        outputs[len(T) / 2], outputs[len(T) / 2 + 1] = outputs[len(T) / 2 + 1], outputs[len(T) / 2]
        rev_outputs = outputs[::-1]

        c_index = get_weighted_C_index(T, outputs)
        ord_error = weighted_c_index_error(T, outputs) / len(T)
        test_error = 1 / c_index
        print("1_off_middle_error = ", ord_error, "test value = ", test_error, "weighted_index = ", c_index)
        assert(ord_error == test_error)

        assert(ord_error > 1)
        assert(ord_error < ord_error_ends)

        c_index = get_weighted_C_index(T, rev_outputs)
        rev_error = weighted_c_index_error(T, rev_outputs) / len(T)
        print("1_off_middle_reversed_error = ", rev_error, "weighted_index = ", c_index)

        assert(rev_error > rev_error_ends)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
