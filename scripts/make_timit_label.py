import scipy.io, ipdb, os
from DNNSMM.util.human_sort import atoi, natural_keys
from os import listdir
from os.path import isfile, isdir, join, splitext, basename
from itertools import groupby
import numpy as np

datadir = '/dataset/kaldi/data-fmllr-tri3'
savedir = '/dataset/kaldi/data-fmllr-tri3-edited'
timitdir = '/dataset/timit/TIMIT'

def make_timit_label(which_set, winsize, shift, fmt = 'nosave'):
    assert which_set in ['TRAIN','DEV','TEST']

    # Obtain alignment information from KALDI
    alilist, phnlen, phnids = aliparser(which_set)

    if which_set in ['DEV']:
         which_set_timit = 'TEST'
    else:
         which_set_timit = which_set
        
    # Obtain directory informations from TIMIT
    timitpath = join(timitdir,which_set_timit)
    subdirs = [x[0] for x in os.walk(timitpath) if x[1]==[]]
    uttlist=[]
    for sub in subdirs:
        uttlist.extend([join(sub,x) for x in listdir(sub) if x.endswith('PHN')])
    
    # save to wid
    savename = join(datadir,'ali_mono_true_'+which_set)
    if fmt == 'pln':
        wid = open(savename,'w')
    elif fmt == 'mat':    
        labels = [] 
        states = []
    elif fmt == 'nosave':
         pass
    else:
         raise ValueError('format should be "pln" or "mat"')

    alisum = 0
    for ali in ['FADG0_SI1909']:
    #for ali in alilist:
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
        numpad = int(phnlen[ind] - len(phnlst))
    
        if not numpad==0:
            print '%15s, kaldi length %3d - timit length %3d = %3d' \
                %(ali, phnlen[ind], len(phnlst), numpad)
        if numpad >0:
            phnlst.extend([phnlst[-1]]*numpad)
        elif numpad < 0:
            phnlst = phnlst[:numpad]

        for x,y in zip(phnids[ind],phnlst):
            print x,y
        
        # Write to wid
        phnlst_ = (str(x) for x in phnlst)
        phnstr = ali + ' ' + ' '.join(phnlst_) + '\n'
        if fmt == 'pln':
            wid.write(phnstr)
        elif fmt == 'mat':
            labels.append(phnlst)
            # Assign state
            statelst = state_label(phnlst)
            states.append(statelst)
        alisum += len(phnlst)

    if fmt == 'pln':
        wid.close()
    elif fmt == 'mat':
        assert len(labels) == len(states)
        ipdb.set_trace()
        scipy.io.savemat(savename, \
            {which_set+'_labels':labels,which_set+'_states':states,'utt':alilist}, appendmat=True)
    print '%s is done, total length is %d' % (which_set, alisum)

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

    # If the total number of frames is longer than kaldi, refine it.
    numphndiff = int(len(phnlst)-nframes(int(phns[-2].split(' ')[1])))
    if type=='cut' and numphndiff>0:
        phnlst=phnlst[:-numphndiff]
    
    return phnlst

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
            newphn = 1
        elif phn=='q' and q=='pre':
            newphn = mapper(phmap,prephone)
        elif phn=='q' and q=='post':
            raise NotImplementedError, "Not implemented yet" 
        else:
            newphn = mapper(phmap,phn)
            prephone = phn
        phoneIds.append(int(newphn))
    return phoneIds

def state_label(phnlst):
    simple_phnlst = [(c,len(list(cgen))) for c,cgen in groupby(phnlst)]
    y=[]
    for SegID, tl in simple_phnlst:
        tl=float(tl)
        (fID,mID,eID) = (SegID*3-2, SegID*3-1, SegID*3)
        ( fl, ml, el) = (np.round(tl/3), np.ceil(tl/3), np.floor(tl/3))

        y.extend([fID]*fl)
        y.extend([mID]*ml)
        y.extend([eID]*el)

    y = np.asarray(y) - 1 # label 1:144 goes to 0:143
    return y

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

def mapper(map, item, c1=0, c2=1):
    idx = np.where(map[c1,:]==item)
    return map[c2,idx[0]][0]

def nframestolength(nframes, winsize=400, shift=160):
    return (nframes-1)*shift+winsize

def nframes(tot_len, winsize=400, shift=160, type='floor'):
    if type == 'floor':
        return 1+np.floor((tot_len-winsize)/shift)
    elif type == 'round':
        return np.floor((2*tot_len-winsize)/(2*shift))+1
    else: 
        return 1+np.ceil((tot_len-winsize)/shift)

def nframe_waveform(uttpath):
    with open(uttpath[:-4]+'.TXT') as f:
        line = f.readline()
    return nframes(int(line.split(' ')[1]))

def pdfid_to_state(pdfid):
    return pdfid%3

if __name__=='__main__':
    #which_set = 'DEV'
    winsize = 400
    shift = 160
    fmt = 'mat' # 'mat' for .mat formatting, 'pln' for plain txt
    [make_timit_label(x, winsize, shift, 'nosave') for x in ['DEV','TRAIN','TEST']]
    #[make_timit_label(which_set, winsize, shift, fmt) \
    #    for which_set in ['DEV','TRAIN','TEST']]
