import os, scipy.io, sys
import numpy as np
from os import listdir
from os.path import isfile, join, splitext, basename
from DNNSMM.util.human_sort import *
import cPickle as pkl
import ipdb

alidir = '/home/thkim/libs/kaldi/egs/timit/s5/exp/tri3_ali'
#arkdir = '/home/thkim/libs/kaldi/egs/timit/s5/data-mfcc-deltas'
arkdir = '/home/thkim/libs/kaldi/egs/timit/s5/data-fmllr-tri3'
#arkdir = '/home/thkim/libs/kaldi/egs/timit/s5/data-fmllr-tri3-tmp'
savepath = '/dataset/kaldi/data-fmllr-tri3' 
matpath = '/dataset/kaldi/data-fmllr-tri3-edited'
mdlpath = '/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn_fmllr_timit_tri_l10_n2048_61'

# Using trained DNN model (mdlpath), we can obtain fproped values
def fprop(which_set):
    datadir = join(arkdir, which_set)
    feature_transform = '--feature-transform='+mdlpath+'/final.feature_transform'
    mdlname = mdlpath+'/final.nnet'
    cntfile = '/dataset/kaldi/data-fmllr-tri3-edited/pdf_count_tri_TRAIN_timit_450'
    #cntfile = '/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts'

    datadir = join(datadir,'data')
    #postfix = '' 
    postfix = '--use-gpu=yes' 
    #postfix = '--use-gpu=yes --no-softmax=true' 
    #postfix = '--use-gpu=yes --no-softmax=true --class-frame-counts=%s' % cntfile
    #postfix = '--use-gpu=yes --apply-log=true --class-frame-counts=%s' % cntfile

    # Obtain file name
    arkfiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.ark') ]
    arkfiles.sort(key=natural_keys)

    # Save it into plain text file
    for ind,fname in enumerate(arkfiles):
        arkname =join(datadir,fname)
        savename=join(savepath,mdlpath.split('/')[-1]+'_'+which_set+'.'+str(ind))
        os.system('nnet-forward %s %s %s ark:%s ark,t:%s'\
            %(feature_transform,postfix,mdlname,arkname,savename))    

# Save .ark file to plain text file
# i.e. save kaldi features (e.g. fmllr) into readable format
def arktopln(datadir):
    datadir = join(datadir,'data')
    
    # Obtain file name
    arkfiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.ark') ]
    arkfiles.sort(key=natural_keys)
    
    # Save it into plain text file
    for fname in arkfiles:
        arkname =join(datadir,fname)
        
        sptext = basename(splitext(arkname)[0])
        savetext = splitext(sptext)[0]+'_tmp'+splitext(sptext)[1]
        
        savename=join(savepath,savetext)
        os.system('copy-feats ark:%s ark,t:%s'%(arkname,savename))    

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

def arkplntomat10(which_set):
    datadir = savepath
    key = mdlpath.split('/')[-1]
    arkfiles = [ f for f in listdir(datadir) if key in f and '_'+which_set in f]
    #arkfiles = [ f for f in listdir(datadir) if '_'+which_set in f and 'tmp.' in f]
    arkfiles.sort(key=natural_keys)

    for fname in arkfiles:
        
        uttname = []
        uttlength = []
        feats = []
        
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
       
        varname = key.replace('-','_')+'_'+which_set 
        
        tempvar = {'feats':feats,
            'uttname':uttname,
            'uttlength':uttlength,
            'uttidx':uttidx}

        scipy.io.savemat(join(matpath,fname),{'featinfo':tempvar},appendmat=True)
        #scipy.io.savemat(join(matpath,varname),{'featinfo':tempvar},appendmat=True)
    print 'saved at %s' % join(matpath,varname)
    del feats, uttname, uttlength

# from archive text file, save it into mat file
def arkplntomat(which_set):
    datadir = savepath
    key = mdlpath.split('/')[-1]
    arkfiles = [ f for f in listdir(datadir) if key in f and '_'+which_set in f]
    #arkfiles = [ f for f in listdir(datadir) if '_'+which_set in f and 'tmp.' in f]
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
   
    varname = key.replace('-','_')+'_'+which_set 
    
    tempvar = {'feats':feats,
        'uttname':uttname,
        'uttlength':uttlength,
        'uttidx':uttidx}

    #scipy.io.savemat(join(matpath,fname),{'featinfo':tempvar},appendmat=True)
    scipy.io.savemat(join(matpath,varname),{'featinfo':tempvar},appendmat=True)
    print 'saved at %s' % join(matpath,varname)
    del feats, uttname, uttlength

if __name__=='__main__':
    # from ark to files
    #map(arktopln,[join(arkdir,'train')])
    #[arkplntomat(x) for x in ['train']]
    #map(arktopln,[join(arkdir,'dev'),join(arkdir,'train'),join(arkdir,'test')])
    #map(fprop,[join(arkdir,'dev'),join(arkdir,'test'),join(arkdir,'train')])

    #[alitopln(x,'tri') for x in ['dev','test','train']]
    #[alitopln(x,'mono') for x in ['dev','test','train']]
    
    # from files to mat file
    #[aliplntomat(x,'tri') for x in ['dev','test','train']]
    #[aliplntomat(x,'mono') for x in ['dev','test','train']]
    
    #map(fprop,['test'])
    #[arkplntomat(x) for x in ['test']]

    map(fprop,['train'])
    [arkplntomat10(x) for x in ['train']]
