import os, scipy.io
import numpy as np
from os import listdir
from os.path import isfile, join, splitext, basename
from DNNSMM.utll import human_sort
import cPickle as pkl
import ipdb

smbr = True 
nlayer = 5
nhids = 512
alidir = '/home/thkim/libs/kaldi/egs/timit/s5/exp/tri3_ali'
arkdir = '/home/thkim/libs/kaldi/egs/timit/s5/data-fmllr-tri3'
savepath = '/dataset/kaldi/data-fmllr-tri3' 
matpath = '/dataset/kaldi/data-fmllr-tri3-edited'



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

def fprop(datadir):
    '''
    if smbr==True:
        feature_transform = '--feature-transform=/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_smbr_'+str(nlayer)+'/final.feature_transform'
        mdlname = '/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_smbr_'+str(nlayer)+'/final.nnet'
    else:
        feature_transform = '--feature-transform=/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_'+str(nlayer)+'/final.feature_transform'
        mdlname = '/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_'+str(nlayer)+'/final.nnet'
    '''
    feature_transform = '--feature-transform=/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_smbr_6_512/final.feature_transform'
    mdlname = '/home/thkim/libs/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_smbr_6_512/final.nnet'
    
    datadir = join(datadir,'data')
    postfix = '--use-gpu=yes'

    # Obtain file name
    arkfiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.endswith('.ark') ]
    arkfiles.sort(key=natural_keys)
    
    # Save it into plain text file
    for fname in arkfiles:
        arkname =join(datadir,fname)
        savename=join(savepath,basename(splitext(arkname)[0]))
        if smbr==True:
            savename=splitext(savename)[0]+'_fprop_l'+str(nlayer)+'_'+str(nhids)+'_smbr'+splitext(savename)[1]
        else:
            savename=splitext(savename)[0]+'_fprop_l'+str(nlayer)+'_'+str(nhids)+splitext(savename)[1]
        os.system('nnet-forward %s %s %s ark:%s ark,t:%s'\
            %(feature_transform,postfix,mdlname, arkname,savename))    

# Save .ali file to plain text file
def saveali(which_set, nmodel):
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

def aliparser(which_set,nmodel):
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

def arkparser(which_set,nmodel):
    datadir = savepath
    arkfiles = [ f for f in listdir(datadir) \
        if isfile(join(datadir,f)) and f.startswith('feats') \
        and which_set in f and 'fprop_l'+str(nlayer)+'_'+str(nhids) in f]
    '''
    # Obtain file name
    if nmodel=='tri':
        arkfiles = [ f for f in listdir(datadir) \
            if isfile(join(datadir,f)) and f.startswith('feats') \
            and which_set in f and 'fprop_l'+str(nlayer) in f]
    else:
        arkfiles = [ f for f in listdir(datadir) \
            if isfile(join(datadir,f)) and f.startswith('feats') \
            and which_set in f and not 'fprop' in f]
    '''
    if smbr==True:
        arkfiles = [ f for f in arkfiles if 'smbr' in  f]
    else:
        arkfiles = [ f for f in arkfiles if not 'smbr' in  f]
    
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
    
    varname = which_set+'_'+nmodel+'_features_l'+str(nlayer)+ '_'+str(nhids)
    
    if smbr == True:
        varname +='_smbr'
    
    tempvar = {'feats':feats,
        'uttname':uttname,
        'uttlength':uttlength,
        'uttidx':uttidx}

    label = alignment(tempvar,which_set,nmodel)
    tempvar['phoneIds']=label
    scipy.io.savemat(join(matpath,varname),{varname:tempvar},appendmat=True)
    del feats, uttname, uttlength

def alignment(var,which_set,nmodel):
    alignment=scipy.io.loadmat(join(matpath,which_set+'_'+nmodel+'_alignment.mat'))
    [uttidx_ali, phoneids, uttname_ali, uttlength_ali]= \
        alignment[which_set+'_'+nmodel+'_alignment'][0][0] # phoneids
    
    # uttname
    uttname_ali = list([utt.strip() for utt in uttname_ali])

    label = []
    for featind, utt in enumerate(var['uttname']):
        aliind = uttname_ali.index(utt)
        ids=phoneids[uttidx_ali[aliind][0]:uttidx_ali[aliind][1]]
        label.extend(ids)
    return label

if __name__=='__main__':
    # from ark to files
    #map(saveark,[join(arkdir,'dev'),join(arkdir,'train'),join(arkdir,'test')])
    #map(fprop,[join(arkdir,'dev'),join(arkdir,'test'),join(arkdir,'train')])
    #map(fprop,[join(arkdir,'dev'),join(arkdir,'test')])

    #[saveali(x,'tri') for x in ['dev','test','train']]
    #[saveali(x,'mono') for x in ['dev','test','train']]
    
    # from files to mat file
    #[aliparser(x,'tri') for x in ['dev','test','train']]
    #[aliparser(x,'mono') for x in ['dev','test','train']]
    
    # fprop features - tri
    [arkparser(x,'tri') for x in ['dev','test']]
    #[arkparser(x,'mono') for x in ['dev','test']]
    #[arkparser(x,'mono') for x in ['train']]

