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

        filedir = '/dataset/data_htk_hs' 
        filename = self.which_set + self.extension
        self.filepath = os.path.join(filedir,filename)
        
        # Load data
        self.load_data()
        self.nData = self.mfcc.shape[1]

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
            [self.mfcc,             # phone MFCC (MFCC dim, # of frames)
            self.phones_id_timit,   # phonesID (# of frames, 1) min = 1, max = 59
            self.phones_id_48,      # phonesID (1, # of frames) min = 1, max = 48
            self.phones_id_39,      # phonesID (1, # of frames) min = 1, max = 39
            self.phonesmap39,       # phonesmap from 48 to 39 (1, 48)
            self.phonesList,        # phoneslist (62,3)
            self.phonesmap48,       # phonesmap from 62 to 48 (1,62)
            self.seg,               # segmentation (# of segs, [first, end])
            self.gender,            # gender of each segmentation (# of segs, 1)
            self.phoneSegID48,      # phone segmentation ID for 48 (1, # of 48segs)
            self.phoneSeg48,        # phone segmentation for 48 (# of 48segs, [first, end])
            self.phoneSegID39,      # phone segmentation ID for 39 (1, # of 39segs)
            self.phoneSeg39,        # phone segmentation for 39 (# of 39segs, [first, end])
            self.segtoSeg48,        # segmentation to segmentation for 48 (# of segs, # of 48segs)
            ] = data[self.which_set+'Data'][0][0]

        elif self.which_set == 'dev':
            [self.mfcc,             # phone MFCC (MFCC dim, # of frames)
            self.phones_id_timit,   # phonesID (# of frames, 1) min = 1, max = 59
            self.phones_id_48,      # phonesID (1, # of frames) min = 1, max = 48
            self.phones_id_39,      # phonesID (1, # of frames) min = 1, max = 39
            self.phonesmap39,       # phonesmap from 48 to 39 (1, 48)
            self.phonesList,        # phoneslist (62,3)
            self.phonesmap48,       # phonesmap from 62 to 48 (1,62)
            self.seg,               # segmentation (# of segs, [first, end])
            self.gender,            # gender of each segmentation (# of segs, 1)
            self.phoneSegID48,      # phone segmentation ID for 48 (1, # of 48segs)
            self.phoneSeg39,        # phone segmentation for 39 (# of 48segs, [first, end])
            self.phoneSegID39,      # phone segmentation ID for 39 (1, # of 39segs)
            self.phoneSeg48,        # phone segmentation for 48 (# of 39segs, [first, end])
            ] = data[self.which_set+'Data'][0][0]

        elif self.which_set == 'test':
            [self.mfcc,             # phone MFCC (MFCC dim, # of frames)
            self.phones_id_timit,   # phonesID (# of frames, 1) min = 1, max = 59
            self.phones_id_48,      # phonesID (1, # of frames) min = 1, max = 48
            self.phones_id_39,      # phonesID (1, # of frames) min = 1, max = 39
            self.phonesmap39,       # phonesmap from 48 to 39 (1, 48)
            self.phonesList,        # phoneslist (62,3)
            self.phonesmap48,       # phonesmap from 62 to 48 (1,62)
            self.seg,               # segmentation (# of segs, [first, end])
            self.gender,            # gender of each segmentation (# of segs, 1)
            self.phoneSeg48,        # phone segmentation for 48 (# of 48segs, [first, end])
            self.phoneSegID48,      # phone segmentation ID for 48 (1, # of 48segs)
            ] = data[self.which_set+'Data'][0][0]

    def feature_design(self):
        tempX = self.mfcc.T
       
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
        y = []
        for ind, SegID in enumerate(self.phoneSegID48[0]):
            tl=float(self.phoneSeg48[ind][1]-self.phoneSeg48[ind][0]+1)
            (fID,mID,eID) = (SegID*3-2, SegID*3-1, SegID*3)
            ( fl, ml, el) = (np.round(tl/3), np.ceil(tl/3), np.floor(tl/3))

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

