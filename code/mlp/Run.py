import os, sys, ipdb, time
import pylearn2
from pylearn2.config import yaml_parse

dirname=os.path.abspath(os.path.dirname(__file__))

def get_hparams(fname,train):
    train_stop = 'None'
    test_start = 0#880001
    test_stop = 'None'#990000
    valid_start = 0#990001
    valid_stop = 'None'
    dim_h0 = 1024 
    max_epochs = 1000
    framesize = 11

    postfix = raw_input("Any postfix? ")
    if not postfix == "":
         postfix = "_"+postfix

    save_path = 'result/'
    save_path = save_path + \
        "_".join([time.strftime("%m%d"),fname,"h"+str(dim_h0)]) + postfix
    
    if os.path.exists(save_path+'.pkl'):
         if 'N'==raw_input('Same exp. exists, continue?([y]/N)'):
             return
    if 'kaldi' in fname:
        hparams = {
            'train_stop': train_stop,
            'valid_start': valid_start,
            'valid_stop': valid_stop,
            'test_start': test_start,
            'test_stop': test_stop,
            'nvis': 40*framesize,
            'dim_h0': dim_h0,
            'framesize': framesize,
            'max_epochs': max_epochs,
            'save_path': save_path}
    else:
        hparams = {
            'train_stop': train_stop,
            'valid_start': valid_start,
            'valid_stop': valid_stop,
            'test_start': test_start,
            'test_stop': test_stop,
            'nvis': 39*framesize,
            'dim_h0': dim_h0,
            'framesize': framesize,
            'max_epochs': max_epochs,
            'save_path': save_path}
    return hparams

for arg in sys.argv:
    fname,ext = os.path.splitext(arg)
    if ext != '.yaml':
        continue
    with open(os.path.join(dirname,arg),'r') as f:
        train= f.read()
    hparams = get_hparams(fname,train)
    savename=hparams['save_path']+'.yaml'
    hparams['best_save_path'] = hparams['save_path'] + '_best.pkl'
    hparams['save_path'] = hparams['save_path'] + '.pkl'

    train = train % (hparams)
    print train
    
    fp = open(savename,'w')
    fp.write(train)
    fp.close()
    
    train_loop = yaml_parse.load(train)
    train_loop.main_loop()
    
    print hparams['save_path']

