import time
from make_timit_label import *

treepath = '/dataset/thkim/data/2014_smm/treemap.mat'
datapath = '/dataset/kaldi/data-fmllr-tri3' 

def tree_pdf(which_set,nmodel='tri',fmt='pln'):
    treemdl = scipy.io.loadmat(treepath)
    if nmodel == 'tri':
         treemap = 'treemap'
    elif nmodel == 'mono':
         treemap = 'treemap_mono'
    tree = np.asarray(treemdl[treemap])

    filepath = join(datapath,'ali_mono_true_'+which_set+'.mat')
    seq = scipy.io.loadmat(filepath)
    labels = [np.asarray(x[0]) for x in seq[which_set+'_labels'][0]]
    states = [np.asarray(x[0]) for x in seq[which_set+'_states'][0]]
    uttlist= [str(x).strip() for x in seq['utt']]

    query_seq = [make_tri_labels(x,y,nmodel) for x,y in zip(labels,states)]
    #pdfids = [query_to_pdf(seq, tree) for seq in query_seq]
    
    uttlength = len(labels)
    pdfids = []

    #ind1 =uttlist.index('FLHD0_SI1974')
    #print ind1
    #for ind in [ind1]:
    for ind in range(uttlength):
        print 'Working on ', uttlist[ind]
        lab = labels[ind]
        sts = states[ind]
        pdfids.append(query_to_pdf(query_seq[ind],tree))

    
    if fmt == 'mat'
    savename = join(datapath,'ali_tri_true_'+which_set)
    scipy.io.savemat(savename,{which_set+'_alignment':pdfids},appendmat=True)
    elif fmt == 'pln': 
        wid = open(join(datapath,'ali_tri_true_'+which_set),'w')
        for utt, id in zip(uttlist,pdfids):
            id_ = (str(x) for x in id)
            phnstr = utt + ' ' + ' '.join(id_) + '\n'
            wid.write(phnstr)
        wid.close()

def make_tri_labels(seq,states,nmodel):
    simple_seq = [(c,len(list(cgen))) for c,cgen in groupby(seq)]
    phnlst = [x[0] for x in simple_seq]
    phnlst_l = [x[1] for x in simple_seq]
    del simple_seq

    phnlst0 = np.concatenate(([1],[1],phnlst))
    phnlst1 = np.concatenate(([1],phnlst,[1])) 
    phnlst2 = np.concatenate((phnlst,[1],[1]))
    tri_phnlst = np.vstack((phnlst0.T,phnlst1.T,phnlst2.T)).T[1:-1]
    ind = np.where(tri_phnlst[:,1]==1)
    tri_phnlst[ind,0]=0
    tri_phnlst[ind,2]=0
    
    tri_seq = []
    for l, triphn in zip(phnlst_l, tri_phnlst):
        tri_seq.extend([triphn]*l)
    tri_seq = np.asarray(tri_seq)

    states = pdfid_to_state(states)
    states = np.expand_dims(states,axis=1)
    query_seq = np.concatenate((states,tri_seq),axis=1)
    return query_seq

def query_to_pdf(query_seq,tree):
    query_seq_tup = [tuple(x) for x in query_seq]
    tree_ind      = [x[0] for x in tree]
    tree_tup      = [tuple(x[1:]) for x in tree]
    tree_dict     = dict(zip(tree_tup,tree_ind))
    tree_tup_set  = set([tuple(x[1:]) for x in tree])
    
    pdfid = []
    for qidx, query in enumerate(query_seq_tup):
        if query in tree_tup_set:
            ind = tree_dict[query]
        else:
            print query, 'does not exists'
            if query_seq[qidx][0]==0:
                ind = subset(tree,query,[3])
                if not ind:
                     ind = subset(tree,query,[1,3])
            elif query_seq[qidx][0]==1:
                ind = subset(tree,query,[1])
                if not ind:
                     ind = subset(tree,query,[1,3])
            else:
                ind = subset(tree,query,[1])
                if not ind:
                     ind = subset(tree,query,[1,3])
            ind = min(ind)

        pdfid.extend([tree_ind[ind]])
    return pdfid

def subset(whole, part, ignore):
    whole_set = set(range(len(whole)))
    for i in range(4):
        if i in ignore:
            continue
        l = np.where(whole[:,i+1]==part[i])
        whole_set=whole_set.intersection(l[0])
    output = [x for x in iter(whole_set)]
    return output

if __name__=='__main__':
    fmt = 'mat'
    [tree_pdf(x,'tri',fmt) for x in ['DEV']]
    #[tree_pdf(x,'tri') for x in ['DEV','TRAIN','TEST']]
