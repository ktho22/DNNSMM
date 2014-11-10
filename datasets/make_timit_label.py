import scipy.io, ipdb, os
from DNNSMM.util.human_sort import atoi, natural_keys
from os import listdir
from os.path import isfile, isdir, join, splitext, basename
import numpy as np

datadir = '/dataset/kaldi/data-fmllr-tri3'
savedir = '/dataset/kaldi/data-fmllr-tri3-edited'
timitdir = '/dataset/timit/TIMIT'

def make_timit_label(which_set, winsize, shift):
    assert which_set in ['TRAIN','DEV','TEST']

    # Obtain alignment information from KALDI
    alilist, phnlen, phnids = aliparser(which_set)

    if which_set in ['TEST','DEV']:
         which_set_timit = 'TEST'
        
    # Obtain directory informations from TIMIT
    timitpath = join(timitdir,which_set_timit)
    subdirs = [x[0] for x in os.walk(timitpath) if x[1]==[]]
    uttlist=[]
    for sub in subdirs:
        uttlist.extend([join(sub,x) for x in listdir(sub) if x.endswith('PHN')])
    
    # save to wid
    wid = open(join(datadir,'ali_mono_true_'+which_set),'w')

    for ali in alilist:
        spk, utt = ali.split('_') 
        uttpath =filter(lambda x:spk in x and utt+'.PHN' in x, uttlist)[0]
        with open(uttpath,'r') as fid:
            phns = fid.read().split('\n')

        # mapping from phns to phnlst
        phnlst60 = make_phonelist(phns,winsize,shift)
        phnlst48 = phone60to48(phnlst60)
        phnlst = phone48toidx(phnlst48,'amb')
        
        # Compare to the length of waveform, refine the length of labels
        ind = alilist.index(ali)
        wav_nframe = int(nframe_waveform(uttpath))
        numpad = int(wav_nframe - len(phnlst))
    
        if numpad >0:
            phnlst.extend([phnlst[-1]]*numpad)
            print '%15s, length diff is %2d' %(ali, numpad)
        elif numpad < 0:
            phnlst = phnlst[:numpad]
            print '%15s, length diff is %2d' %(ali, numpad)

        # Write to wid
        phnlst_ = (str(x) for x in phnlst)
        phnstr = ali + ' ' + ' '.join(phnlst_) + '\n'
        wid.write(phnstr)
    wid.close()

def make_phonelist(phns,winsize,shift,type='cut'):
    phnlst= []
    remaining = 0
    for ind, phn in enumerate(phns):
        if phn=='':
             continue
        phn = phn.split(' ')
        
        if ind==0 and not int(phn[0])==0:
            phn[0]=0
             
        rawphnlength = int(phn[1])-int(phn[0])
        phnlength = rawphnlength + remaining 
        numphn = nframes(phnlength, winsize, shift, type='round')
        truelength = nframestolength(numphn,winsize,shift)
        remaining = phnlength - truelength + (winsize - shift)
        phnlst.extend([phn[2]]*numphn)

    # If the total number of frames is longer than size, refine it.
    numphndiff = int(len(phnlst)-nframes(int(phns[-2].split(' ')[1])))
    if type=='cut' and numphndiff>0:
        phnlst=phnlst[:-numphndiff]
    
    return phnlst

def nframestolength(nframes, winsize=400, shift=160):
    return (nframes-1)*shift+winsize

def nframes(tot_len, winsize=400, shift=160, type='floor'):
    if type == 'floor':
        return 1+np.floor((tot_len-winsize)/shift)
    elif type == 'round':
        return np.floor((2*tot_len-winsize)/(2*shift))+1
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

def nframe_waveform(uttpath):
    with open(uttpath[:-4]+'.TXT') as f:
        line = f.readline()
    return nframes(int(line.split(' ')[1]))

if __name__=='__main__':
    which_set = 'DEV'
    winsize = 400
    shift = 160

    make_timit_label(which_set, winsize, shift)
