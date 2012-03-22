from kalderstam.util.filehandling import parse_file, save_network, load_network
from kalderstam.neural.network import build_feedforward, build_feedforward_committee
import logging
import survival.cox_error as cox_error
from kalderstam.neural.training.gradientdescent import traingd

def test():
    #numpy.seterr(all = 'raise')

    p = 4 #number of input covariates

    #net = build_feedforward(p, 8, 1, output_function = "linear")
    net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/4x10x10x1.ann')
    #net = load_network('/home/jonas/workspace/aNeuralN/ANNs/4x10x10x1.ann')

    #filename = '/home/jonas/Dropbox/ANN/my_tweaked_fake_data_no_noise.txt'
    #filename = '/home/gibson/jonask/my_tweaked_fake_data_no_noise.txt'
    #filename = '/home/gibson/jonask/my_tweaked_fake_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_no_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_no_noise.txt'

    #P, T = parse_file(filename, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
    P, T = parse_file(filename, targetcols = [4, 5], inputcols = [-1, -2, -3, -4], ignorerows = [0], normalize = True)

    try:
        net = traingd(net, (P, T), (None, None), epochs = 10, learning_rate = 5, block_size = 100, error_module = cox_error)
    except FloatingPointError:
        print('Aaawww....')

#This is a test of the functionality in this file
if __name__ == '__main__':

    import pstats, cProfile

    #numpy.seterr(all = 'raise')
    logging.basicConfig(level = logging.INFO)

    cProfile.runctx("test()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

    #test()
    #plt.show()
