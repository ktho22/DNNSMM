import scipy.io, ipdb, os
from DNNSMM.util.human_sort import atoi, natural_keys
from os import listdir
from os.path import isfile, isdir, join, splitext, basename
import numpy as np

timitdir = '/dataset/timit/TIMIT'

def make_timit_label(which_set, winsize, shift):
    datadir = join(timitdir,which_set)
    subdirs = [x[0] for x in os.walk(datadir) if x[1]==[]]
    subdirs.sort(key=natural_keys)

    
    for sub in subdirs:
        uttlist = [x for x in listdir(sub) if x.endswith('PHN')]
        uttlist.sort(key=natural_keys)
        spkname = basename(sub)
        for utt in uttlist:
            uttid = spkname+'_'+splitext(utt)[0]
            uttpath = join(sub,utt)
            with open(uttpath,'r') as fid:
                phns = fid.read().split('\n')
            phnlst60 = make_phonelist(phns,winsize,shift)
            phnlst48 = phone60to48(phnlst60)
            phnlst = phone48toidx(phnlst48,'pre')
            phnlst = (str(x) for x in phnlst)
            phnstr = uttid + ' ' + ' '.join(phnlst)


def make_phonelist(phns,winsize,shift,type='cut'):
    phnlst= []
    remaining = 0
    for phn in phns:
        if phn=='':
             continue
        phn = phn.split(' ')
        rawphnlength = int(phn[1])-int(phn[0])
        phnlength = rawphnlength + remaining 
        numphn = nframes(phnlength, winsize, shift, type='round')
        truelength = nframestolength(numphn,winsize,shift)
        remaining = phnlength - truelength + (winsize - shift)
        phnlst.extend([phn[2]]*numphn)
    
    if nframestolength(len(phnlst),winsize,shift) and type == 'cut':
        phnlst = phnlst[:-1]
    return phnlst

def nframestolength(nframes, winsize, shift):
    return (nframes-1)*shift+winsize

def nframes(tot_len, winsize, shift, type='floor'):
    if type == 'floor':
        return 1+np.floor((tot_len-winsize)/shift)
    elif type == 'round':
        return 1+np.round((tot_len-winsize)/shift)
    else: 
        return 1+np.ceil((tot_len-winsize)/shift)

def phone60to48(phone60):
    mapfile = '/home/thkim/libs/kaldi/egs/timit/s5/conf/phones.60-48-39.map'
    with open(mapfile,'r') as fid:
        prephmap = fid.read().split('\n')
    phmap = []
    for item in prephmap:
        if item == '':
             continue
        elif item == 'q':
            phmap.append(['q','q','q'])
        else:
            phmap.append(item.split('\t'))
    phmap = np.asarray(phmap).T
    
    phone48=[]
    for phn in phone60:
        newphn = mapper(phmap,phn)
        phone48.append(newphn)
    return phone48

def phone48toidx(phone48,q='amb'):
    mapfile = '/home/thkim/libs/kaldi/egs/timit/s5/data/lang/phones.txt'
    with open(mapfile,'r') as fid:
        prephmap = fid.read().split('\n')
    phmap = []
    for item in prephmap:
        if item == '':
             continue
        phmap.append(item.split(' '))
    phmap = np.asarray(phmap).T
   
    prephone=''
    phoneIds=[]
    for ind, phn in enumerate(phone48):
        if phn=='q' and q=='amb':
            newphn = 49 
        elif phn=='q' and q=='pre':
            newphn = mapper(phmap,prephone)
        elif phn=='q' and q=='post':
            raise NotImplementedError, "Not implemented yet" 
        else:
            newphn = mapper(phmap,phn)
            prephone = phn
        phoneIds.append(newphn)
    return phoneIds

def mapper(map, item, c1=0, c2=1):
    idx = np.where(map[c1,:]==item)
    return map[c2,idx[0]][0]

if __name__=='__main__':
    which_set = 'TRAIN'
    winsize = 400
    shift = 160

    make_timit_label(which_set, winsize, shift)
