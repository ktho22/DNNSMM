import theano, pylearn2
import theano.tensor as T
import cPickle as pkl
import sys, scipy.io, os, ipdb
#from DNNSMM.datasets.timit import TIMIT
from DNNSMM.datasets.kaldi_timit import TIMIT
import numpy as np

def genFeat(fname):
    # re-generate file names (assumption: the ext. of fname is .pkl)
    savepath = '/nas/members/THKIM/2014_SMM/feat/'
    basename = os.path.splitext(fname)[0]
    matname = savepath + os.path.basename(basename) + '.mat'
    if 'best' in basename:
        yamlname = '_'.join(basename.split('_')[:-1]) + '.yaml'
    else:
        yamlname = basename + '.yaml'

    # Load model and data
    mdl = pkl.load(open(fname))
    testdata = TIMIT('test', 0, None, framesize=11)
    devdata = TIMIT('dev', 0, None, framesize=11)
    
    # Make tensor variables and compile graph
    x = mdl.get_input_space().make_batch_theano()
    fprop = theano.function([x],mdl.fprop(x))

    # Compute features from fprop and save into .mat file
    testScores = np.log(fprop(testdata.X)).T
    devScores = np.log(fprop(devdata.X)).T
    feat = {'testScores': testScores, 'devScores': devScores}
    scipy.io.savemat(matname,feat)
    print 'saved at %s' % matname

if __name__=='__main__':
    assert len(sys.argv)>1, "input argument!"
    map(genFeat,sys.argv[1:])
