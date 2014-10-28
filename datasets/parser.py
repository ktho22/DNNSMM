import os, scipy.io
import numpy as np
from os import listdir
import cPickle as pkl
from os.path import isfile, join, splitext, basename
import re
import ipdb

alidir = '/home/thkim/libs/kaldi/egs/timit/s5/exp/tri3_ali'
arkdir = '/home/thkim/libs/kaldi/egs/timit/s5/data-fmllr-tri3'
kaldidir = '~/libs/kaldi'
savepath = '/dataset/kaldi/data-fmllr-tri3' 
matpath = '/dataset/kaldi/data-fmllr-tri3-edited'

# Utils
def atoi(text):
    return int(text) if text.isdigit() else text

# Utils
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# Save .ark file to plain text file
def saveark(datadir):
    datadir = join(datadir,'data')
    
    # Obtain file name
    arkfiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.ark') ]
    arkfiles.sort(key=natural_keys)
    
    # Save it into plain text file
    for fname in arkfiles:
        arkname =join(datadir,fname)
        savename=join(savepath,basename(splitext(arkname)[0]))
        os.system('copy-feats ark:%s ark,t:%s'%(arkname,savename))    

# Save .ali file to plain text file
def savetriali(which_set):
    datadir = alidir+'_'+which_set
    mdlname = join(datadir,'final.mdl')
    
    # Obtain file name
    alifiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.gz') and f.startswith('ali.')]
    alifiles.sort(key=natural_keys)

    # Save it into plain text file
    for fname in alifiles:
        aliname =join(datadir,fname)
        bname= basename(splitext(aliname)[0])
        sname = splitext(bname)[0]+'_tri_'+which_set+splitext(bname)[1]
        savename=join(savepath,sname)
        os.system('ali-to-pdf %s "ark:gunzip -c %s|" ark,t:%s'\
            %(mdlname,aliname,savename))    

# Save .ali file to plain text file
def saveali(which_set, nmodel):
    datadir = alidir+'_'+which_set
    mdlname = join(datadir,'final.mdl')
    assert nmodel in ['mono','tri']
    cmd = {'mono': 'ali-ti-phones --per-frame', 'tri': 'ali-to-pdf'}[nmodel]

    # Obtain file name
    alifiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.gz') and f.startswith('ali.')]
    alifiles.sort(key=natural_keys)

    # Save it into plain text file
    for fname in alifiles:
        arkname =join(datadir,fname)
        bname= basename(splitext(arkname)[0])
        sname = splitext(bname)[0]+'_'+which_set+splitext(bname)[1]
        savename=join(savepath,sname)
        os.system('%s %s "ark:gunzip -c %s|" ark,t:%s'\
            %(cmd, mdlname,arkname,savename))    

def aliparser(which_set,nmodel):
    datadir = savepath
    keyword = {'mono': 'ali_'+which_set, 'tri': 'ali_tri_'+which_set}[nmodel]

    # Obtain file name
    alifiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.startswith(keyword)]
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

def arkparser(which_set,nmodel):
    datadir = savepath
    # Obtain file name
    arkfiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.startswith('feats') and which_set in f]
    arkfiles.sort(key=natural_keys)

    uttname = []
    uttlength = []
    feats = []

    for fname in arkfiles:
        print 'file %s is processing ...' % fname
        rid = open(join(savepath,fname),'r')
        cnt = 0 
        line = rid.readline()
        while not line in [[], '']:
            spline = line.split()
            if spline[-1]=='[':
                uttname.append(spline[0])
            elif spline[-1]==']':
                feat = map(float,spline[:-1])
                feats.append(feat)
                uttlength.append(cnt+1)
                cnt=0
            else:
                cnt += 1     
                feat = map(float,spline)
                feats.append(feat)
            line = rid.readline()
        rid.close()

    uttendidx = np.cumsum(uttlength)
    uttstidx = [0]
    uttstidx.extend(uttendidx[:-1]+1)
    uttidx = np.asarray([uttstidx,uttendidx]).T

    varname = which_set+'_'+nmodel+'_features'
    tempvar = {'feats':feats,
        'uttname':uttname,
        'uttlength':uttlength,
        'uttidx':uttidx}

    label = alignment(tempvar,which_set)
    tempvar['phoneIds']=label
    scipy.io.savemat(join(matpath,varname),{varname:tempvar},appendmat=True)
    ipdb.set_trace() 
    del feats, uttname, uttlength

def alignment(var,which_set):
    alignment=scipy.io.loadmat(join(matpath,which_set+'_alignment.mat'))
    [uttidx_ali, phoneids, uttname_ali, uttlength_ali]= \
        alignment[which_set+'_alignment'][0][0] # phoneids
    
    # uttname
    uttname_ali = list([utt.strip() for utt in uttname_ali])

    label = []
    for featind, utt in enumerate(var['uttname']):
        aliind = uttname_ali.index(utt)
        ids=phoneids[uttidx_ali[aliind][0]:uttidx_ali[aliind][1]]
        label.extend(ids)
    return label

if __name__=='__main__':
    nmodel = 'mono' # tri or mono

    # from ark to files
    #map(saveark,[join(arkdir,'dev'),join(arkdir,'train'),join(arkdir,'test')])
    #[saveali(x,nmodel) for x in ['dev','test','train']]
    
    # from files to mat file
    [aliparser(x,nmodel) for x in ['dev','test','train']]
    [arkparser(x,nmodel) for x in ['dev','test','train']]

