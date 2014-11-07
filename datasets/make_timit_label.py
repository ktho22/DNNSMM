import scipy.io, ipdb, os
from DNNSMM.util.human_sort import atoi, natural_keys
from os import listdir
from os.path import isfile, isdir, join, splitext, basename
import numpy as np

datadir = '/dataset/kaldi/data-fmllr-tri3'
savedir = '/dataset/kaldi/data-fmllr-tri3-edited'
timitdir = '/dataset/timit/TIMIT'

def make_timit_label(which_set, winsize, shift):
    datadir = join(timitdir,which_set)
    subdirs = [x[0] for x in os.walk(datadir) if x[1]==[]]
    alilist, phnlen, phnids = aliparser(which_set)

    uttlist=[]
    for sub in subdirs:
        uttlist.extend([join(sub,x) for x in listdir(sub) if x.endswith('PHN')])
    
   # wid = open(join(datadir,'ali_mono_true'+which_set),'w')
    wid = open('lengthinfo','w')

    for ali in [alilist[8]]:
    #for ali in alilist:
        spk, utt = ali.split('_') 
        uttpath =filter(lambda x:spk in x and utt in x, uttlist)[0]
        with open(uttpath,'r') as fid:
            phns = fid.read().split('\n')
        phnlst60 = make_phonelist(phns,winsize,shift)
        phnlst48 = phone60to48(phnlst60)
        phnlst = phone48toidx(phnlst48,'pre')

        ind = alilist.index(ali)
        
        numpad =  phnlen[ind] -len(phnlst) 
        #phnlst.append(phnlst[-1]*numpad)
        
        length_ori = phnlen[ind]
        length = len(phnlst)
        ipdb.set_trace()
        print ali, length_ori, length, length_ori-length
        wid.write(' '.join((ali,str(length_ori),str(length),str(numpad)))+'\n')

        #print [[x,y] for x,y in zip(phnids[ind], phnlst)]

        phnlst_ = (str(x) for x in phnlst)
        phnstr = ali + ' ' + ' '.join(phnlst_) + '\n'
        #wid.write(phnstr)
    wid.close()

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
    
    if nframestolength(len(phnlst),winsize,shift)>phns[-2].split(' ')[1]\
        and type == 'cut':
        phnlst = phnlst[:-1]
    

    return phnlst

def nframestolength(nframes, winsize, shift):
    return (nframes-1)*shift+winsize

def nframes(tot_len, winsize, shift, type='floor'):
    if type == 'floor':
        return 1+np.floor((tot_len-winsize)/shift)
    elif type == 'round':
        n_in = 1+np.floor((tot_len-winsize)/shift)
        l_res = tot_len-nframestolength(n_in,winsize,shift)
        n_res = max(0,np.floor((2*l_res-winsize)/(2*shift)))
        return n_in+n_res
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

def aliparser(which_set,nmodel='mono'):
    keyword = 'ali_' + nmodel

    # Obtain file name
    alifiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.startswith(keyword) and which_set.lower() in f]
    alifiles.sort(key=natural_keys)

    uttname = []
    phnids = []
    phnlen = []
    for fname in alifiles:
        rid = open(join(datadir,fname),'r')
        line = rid.readline()
        while not line in [[], '']:
            spline = line.split()
            uttname.append(spline[0].strip())
            phnids.append(map(int,spline[1:]))
            phnlen.append(len(spline[1:]))
            line = rid.readline()
        rid.close()
    return uttname, phnlen, phnids

if __name__=='__main__':
    which_set = 'TRAIN'
    winsize = 400
    shift = 160

    make_timit_label(which_set, winsize, shift)
