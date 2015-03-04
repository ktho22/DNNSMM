'''
This function is for modify extension of .ali files to plan text or .mat files

How to use:
   1. You need to specify nmodel(monophone or triphone), and whichset
'''
import os, scipy.io, sys
import numpy as np
from os import listdir
from os.path import isfile, join, splitext, basename
from DNNSMM.util.human_sort import *
import cPickle as pkl
import ipdb

alidir = '/home/thkim/libs/kaldi/egs/timit/s5/exp/tri3_ali'
savepath = '/dataset/kaldi/data-fmllr-tri3' 
matpath = '/dataset/kaldi/data-fmllr-tri3-edited'


# Save .ali file to plain text file
def alitopln(which_set, nmodel):
    datadir = alidir+'_'+which_set
    mdlname = join(datadir,'final.mdl')
    assert nmodel in ['mono','tri']
    cmd = {'mono': 'ali-to-phones --per-frame', 'tri': 'ali-to-pdf'}[nmodel]

    # Obtain file name
    alifiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.gz') and f.startswith('ali.')]
    alifiles.sort(key=natural_keys)

    # Save it into plain text file
    for fname in alifiles:
        arkname =join(datadir,fname)
        bname= basename(splitext(arkname)[0])
        sname = splitext(bname)[0]+'_'+nmodel+'_'+which_set+splitext(bname)[1]
        savename=join(savepath,sname)
        os.system('%s %s "ark:gunzip -c %s|" ark,t:%s'\
            %(cmd, mdlname,arkname,savename))    

# from alignment plain text, save it into mat file
def aliplntomat(which_set,nmodel):
    datadir = savepath
    keyword = 'ali_' + nmodel

    # Obtain file name
    alifiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.startswith(keyword) and which_set in f]
    alifiles.sort(key=natural_keys)

    uttname = []
    uttlength = []
    phoneids = []
    for fname in alifiles:
        print 'file %s is processing ...' % fname
        rid = open(join(savepath,fname),'r')
        line = rid.readline()
        while not line in [[], '']:
            spline = line.split()
            uttname.append(spline[0].strip())
            phoneids.extend(map(int,spline[1:]))
            uttlength.append(len(spline)-1)
            line = rid.readline()
        rid.close()
    
    uttendidx = np.cumsum(uttlength)
    uttstidx = [0]
    uttstidx.extend(uttendidx[:-1])
    uttidx = np.asarray([uttstidx,uttendidx]).T

    varname = which_set+'_'+nmodel+'_alignment'
    tempvar = {'phoneids':phoneids,
        'uttname':uttname,
        'uttlength':uttlength,
        'uttidx':uttidx}
    scipy.io.savemat(join(matpath,varname),{varname:tempvar},appendmat=True)

if __name__=='__main__':
    [alitopln(x,'tri') for x in ['dev','test','train']]
    [alitopln(x,'mono') for x in ['dev','test','train']]
    
    # from files to mat file
    [aliplntomat(x,'tri') for x in ['dev','test','train']]
    [aliplntomat(x,'mono') for x in ['dev','test','train']]
