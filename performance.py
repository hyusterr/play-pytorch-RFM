# this script is for generating classification report
# usage: python performance.py path/to/y_true.csv path/to/y_pred.csv path/to/idx_to_label.pickle


import sys
import pickle
import numpy as np
from sklearn.metrics import classification_report


def evaluation( y_true_path, y_pred_path, idx_to_label_path ):

    with open( y_true_path, 'r' ) as f:
        y_true = np.array( int(i) for i in f.readlines() )
        f.close()

    with open( y_pred_path, 'r' ) as f:
        y_pred = np.array( int(i) for i in f.readlines() )

    with open( idx_to_label_path, 'rb' ) as f:
        idx_to_label = pickle.load( f )
        f.close()
    
    indexes = [ idx_to_label[i] for i in range( len( set( y_true ) ) ) ]

    return classification_report( y_true, y_pred, target_names=indexes )


if __name__ = '__main__':
 
    print( evaluation( sys.argv[1], sys.argv[2], sys.argv[3] ) )
