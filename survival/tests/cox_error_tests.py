'''
Created on Apr 11, 2011

@author: jonask
'''
import unittest
import numpy as np
from survival.cox_error import get_risk_groups, \
    calc_sigma, derivative_sigma, shift, derivative_error, calc_beta, \
    get_beta_force, derivative_beta, get_y_force, generate_random_data
from survival.cox_error_in_c import get_C_index
from kalderstam.util.numpyhelp import indexOf
from random import sample

class Test(unittest.TestCase):

    def testCIndex(self):
        T, timeslots = generate_random_data(1000)
        outputs, rndtimeslots = generate_random_data(1000)
        #Should give perfect result
        c_perfect = get_C_index(T, T)
        print(c_perfect)
        assert(c_perfect == 1.0)

        #should give around 0.5, that is random
        c_rnd = get_C_index(T, outputs)
        print(c_rnd)
        assert(c_rnd < 0.6 and c_rnd > 0.4)

        #Make all outputs zero. This should NOT give good result
        outputs = np.zeros_like(outputs)
        c_zero = get_C_index(T, outputs)
        print(c_zero)
        assert(c_zero < 0.6)

    def testCIndexShuffling(self):
        T, timeslots = generate_random_data(1000)
        def right(ar):
            return ar[:, 0] + ar[:, 1] * ar[:, 2]
        def wrong(ar):
            return ar[:, 0] - ar[:, 1] * ar[:, 2]
        all = P = np.random.rand(1000, 7) #Filler column to the right

        all[:, 4] = T[:, 1]
        all[:, 3] = right(all[:, 0:3])
        all[:, 5] = wrong(all[:, 0:3])

        c_all_wrong = get_C_index(all[:, 3:5], all[:, 5:7])

        print("C all wrong = " + str(c_all_wrong))

        all_shuffled = np.copy(all)
        np.random.shuffle(all_shuffled)

        c_all_shuffled = get_C_index(all[:, 3:5], all[:, 5:7])

        print("C all shuffled = " + str(c_all_shuffled))

        assert(c_all_shuffled == c_all_wrong)


    def testGetRiskGroups(self):
        outputs, timeslots = generate_random_data(50)
        risk_groups = get_risk_groups(outputs, timeslots)
        for group, start_index in zip(risk_groups, range(len(timeslots))):
            for ti in timeslots[:start_index + 1]:
                assert(ti in group)
                assert(outputs[ti, 1] == 1)

    def testPartFunc(self):
        outputs, timeslots = generate_random_data(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = generate_random_data(100)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        for z, risk_group in zip(part_func, risk_groups):
            testz = np.sum(np.exp(beta * outputs[risk_group, 0]))
            assert(z == testz)

    def testWeightedAverage(self):
        outputs, timeslots = generate_random_data(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = generate_random_data(100)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        for w, z, risk_group in zip(weighted_avg, part_func, risk_groups):
            testw = 1 / z * np.sum(np.exp(beta * outputs[risk_group, 0]) * outputs[risk_group, 0])
            assert(round(w, 10) == round(testw, 10))


    def testCalc_beta(self):
        """Calculate beta for a predetermined optimal value.
        The more patients you have, the larger the value of Beta will be before it diverges.
        Should actually limit calculations to NO MORE than 100 patients. Even that is above my comfort zone really.
        Something else that determines the magnitude of Beta is the "disorder" in the outputs.
        If two outputs are wrong total, and are "close" in the set, then Beta will be alot larger compared to
        two outputs which are wrong but "far away" in the set. The reason is simply because if they are far apart,
        more risk groups will be affected. If they are close, most risk groups will actually be entirely correct.
        Causing beta to grow."""
        #Check that it diverges if given a perfect ordering
        outputs = np.array([[i, 1] for i in np.linspace(0, 5, 100)])
        timeslots = np.arange(100, dtype = np.int64) #0-99
        risk_groups = get_risk_groups(outputs, timeslots)

        #print outputs
        #print timeslots
        #print risk_groups
        diverged = False
        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError:
            #print("Diverged")
            diverged = True #It should diverge in this case
        assert(diverged)
        #Just change one value, and it should now no longer diverge
        outputs[38], outputs[97] = outputs[97], outputs[38]
        #outputs[8], outputs[3] = outputs[3], outputs[8]

        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError:
            #print("Diverged, when it shouldn't")
            assert()
        #If you want to view the value of beta in different cases of "disorder", uncomment two lines below
        #print(beta)
        #assert()

        #Now test that beta is actually a reasonable results
        #That means that F(Beta) = 0 (or very close to zero at least)
        F_result = 0
        for s in timeslots:
            F_result += outputs[s, 0] - weighted_avg[s]
        assert(round(F_result, 4) == 0)

    def testDerivativeSigma(self):
        """Testing Derivative Sigma"""
        outputs, timeslots = generate_random_data(100)
        sigma = calc_sigma(outputs)
        avg = outputs[:, 0].sum() / len(outputs)
        #First calculate it manually, then compare with function
        for i in xrange(len(outputs)):
            output = outputs[i, 0]
            ds = (output - avg) / (len(outputs) * sigma)
            assert(ds == derivative_sigma(sigma, i, outputs))

    def testDerivativeError(self):
        """Testing Derivative Error"""
        outputs, timeslots = generate_random_data(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = generate_random_data(100)
        #outputs, timeslots = generate_random_data(100)
        sigma = calc_sigma(outputs)
        #risk_groups = get_risk_groups(outputs, timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        testDE = -np.exp(shift - beta * sigma) / (1 + np.exp(shift - beta * sigma))

        assert(testDE == derivative_error(beta, sigma))

    def testYForce(self):
        """Testing Y Force."""
        outputs, timeslots = generate_random_data(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = generate_random_data(100)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        for output_index in xrange(len(outputs)):
            #Do the derivative for every Yi
            output = outputs[output_index, 0]
            test_yforce = 0
            for es, risk_group, z, w in zip(timeslots, risk_groups, part_func, weighted_avg):
                if es == output_index:
                    delta = 1
                else:
                    delta = 0
                if output_index in risk_group:
                    wpart = np.exp(beta * output) / z * (1 + beta * (output - w))
                else:
                    wpart = 0
                test_yforce += delta - wpart

            yforce = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)

            assert(test_yforce == yforce)


    def testBetaForce(self):
        """Testing beta force"""
        outputs, timeslots = generate_random_data(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = generate_random_data(100)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        testbeta_force = 0
        for risk_group, z, w in zip(risk_groups, part_func, weighted_avg):
            exp_value = np.exp(beta * outputs[risk_group, 0])
            exp_value_yi = exp_value * outputs[risk_group, 0]
            exp_value_yi2 = exp_value_yi * outputs[risk_group, 0]

            testbeta_force += 1 / exp_value.sum() * exp_value_yi2.sum() - (exp_value_yi.sum() / exp_value.sum()) ** 2

        testbeta_force *= -1

        betaforce = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        assert(round(testbeta_force, 10) == round(betaforce, 10))

    def testDerivativeBeta(self):
        outputs, timeslots = generate_random_data(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = generate_random_data(100)

        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)
        for output_index in xrange(len(outputs)):
            y_force = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)

            dBdYi = -y_force / beta_force
            method_value = derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)

            assert(round(dBdYi, 10) == round(method_value, 10)) #Otherwise errors might happen on 32-bit machines


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
