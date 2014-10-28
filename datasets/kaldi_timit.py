from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
import scipy.io
import os, theano, ipdb

class TIMIT(DenseDesignMatrix):
    def __init__(self,
                 which_set,
                 start=0,
                 stop=None,
                 framesize=11,
                 framePad=True
                 ):
        if type(stop) == str:
             stop = eval(stop)
        self.__dict__.update(locals())
        self.extension = '.mat'
        del self.self
        assert which_set in ['train','dev','test']

        filedir = '/dataset/kaldi/data-fmllr-tri3-edited' 
        filename = self.which_set + '_features' + self.extension
        self.filepath = os.path.join(filedir,filename)
        
        # Load data
        self.load_data()
        self.nData = self.feats.shape[0]

        # Generate features and labels for each examples  
        self.X = self.feature_design()
        self.y = self.label_design()
        assert self.X.shape[0] == self.y.shape[0]
        
        # select the number of examples
        self.X = self.X[self.start:self.stop]
        self.y = self.y[self.start:self.stop]
        
        super(TIMIT, self).__init__(
            X=self.X, y=self.y, y_labels=144
        )
        
    def load_data(self):
        data = scipy.io.loadmat(self.filepath)
        if self.which_set == 'train':
            [self.uttidx,
            self.phoneids,
            self.uttname,
            self.uttlength,
            self.feats,
            ]=data[self.which_set+'_features'][0][0]

        elif self.which_set == 'dev':
            [self.uttidx,
            self.phoneids,
            self.uttname,
            self.uttlength,
            self.feats,
            ]=data[self.which_set+'_features'][0][0]

        elif self.which_set == 'test':
            [self.uttidx,
            self.phoneids,
            self.uttname,
            self.uttlength,
            self.feats,
            ]=data[self.which_set+'_features'][0][0]

    def feature_design(self):
        tempX = self.feats
       
        if self.framePad:
            # For padding frames
            first_frame_tile = np.tile(tempX[0],(int((self.framesize-1)/2),1))
            last_frame_tile = np.tile(tempX[self.nData-1],(int((self.framesize-1)/2),1))
            tempX = np.vstack((first_frame_tile,tempX,last_frame_tile))
            X = tempX[:self.nData]
            # Concatenate frames
            for i in range(1,self.framesize):
                X = np.concatenate((X,tempX[i:self.nData+i]),axis=1)
        else:
            X = tempX[:self.nData-self.framesize+1]
            # Concatenate frames
            for i in range(1,self.framesize):
                X = np.concatenate((X,tempX[i:self.nData-self.framesize+i+1]),axis=1)
        X = np.asarray(X,dtype=theano.config.floatX)
        # Preprocessing should be placed here

        return X
        
    def label_design(self):
        ids = self.phoneids[0]
        
        self.phoneSegID48=[ids[0]]
        self.phoneSegID48Length=[]
        cnt =1
        for i,v in enumerate(ids[1:]):
            if self.phoneSegID48[-1] == v:
                cnt+=1
                if i == len(ids)-2:
                    self.phoneSegID48Length += [cnt]
            else:
                self.phoneSegID48 += [v]
                self.phoneSegID48Length += [cnt]
                cnt=1
        
        endidx = np.cumsum(self.phoneSegID48Length)
        stidx = [0]
        stidx.extend(endidx[:-1]+1)
        self.phoneSeg48 = np.asarray([stidx,endidx]).T
        self.phoneSegID48 = np.asarray(self.phoneSegID48)

        y = []
        for ind, SegID in enumerate(self.phoneSegID48):
            #tl=float(self.phoneSeg48[ind][1]-self.phoneSeg48[ind][0])
            tl=float(self.phoneSegID48Length[ind])
            (fID,mID,eID) = (SegID*3-2, SegID*3-1, SegID*3)
            ( fl, ml, el) = (np.round(tl/3), np.ceil(tl/3), np.floor(tl/3))
            assert tl == fl + ml + el, 'error!'
            y.extend([fID]*fl)
            y.extend([mID]*ml)
            y.extend([eID]*el)
        
        y = np.asarray(y) - 1 # label 1:144 goes to 0:143
        y = y.reshape(y.shape[0], 1)

        if not self.framePad:
            # length of X should be smaller due to concat_frames
            buff = int((self.framesize-1)/2)
            y = y[buff:self.nData-buff] 
        return y
        
if __name__=='__main__':
     train=TIMIT(which_set='train',framesize=11)
     print 'train is successfully loaded'
     test=TIMIT(which_set='test',framesize=11)
     print 'test is successfully loaded'
     dev=TIMIT(which_set='dev',framesize=11)
     print 'dev is successfully loaded'

