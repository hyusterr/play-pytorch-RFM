# this script is for preprocess data and define dataset class for troch model to use
# usage: python data.py /path/to/orignal_data /path/to/preprocess-data.pickle

import pickle
import numpy as np
import csv
import sys
from sklearn.utils import shuffle
from torch.utils.data import Dataset

def read_csv( path ):
    
    with open( path, 'r' ) as f:
        csv_reader = csv.reader( f )
        csv_list   = [ i for i in csv_reader ]
        f.close()
    
    return csv_list 


def split_X_y( lst ):
    
    header = lst[0]
    X = np.array( [ [ float( d ) for d in i[:-1] ] for i in lst[1:] ] )
    y = [ i[-1] for i in lst[1:] ]

    label_to_idx = { label: index for index, label in enumerate(list(set(y))) }
    idx_to_label = { index: label for label, index in label_to_idx.items() }

    y = np.array( [ label_to_idx[i] for i in y ] )

    return X, y, idx_to_label, header


def split_train_test( X, y ):

    X, y = shuffle( X, y )
    cut = int( len(X) * 0.1 )

    train_X, train_y = X[:cut], y[:cut]
    test_X, test_y   = X[cut:], y[:cut]

    return train_X, train_y, test_X, test_y


class RFMDataset( Dataset ):

    def __init__( self, X, y ):
        self.X = X
        self.y = y

    def __len__( self ):
        return len( self.X )

    def __getitem__( self, idx ):
        return self.X[idx], self.y[idx]
    
    

if __name__ == '__main__':

    print( 'reading data...' )
    data = read_csv( sys.argv[1] )
    
    print( 'splitting X, y...' )
    X, y, idx_to_label, header = split_X_y( data )
    
    print( 'splitting train/test...' )
    train_X, train_y, test_X, test_y = split_train_test( X, y)

    print( 'saving to pickle...' )
    with open( sys.argv[2], 'wb' ) as f:
        pickle.dump( ( train_X, train_y, test_X, test_y, idx_to_label, header ), f)
        f.close()

    print( 'done!' )
