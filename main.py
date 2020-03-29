# this script is for running the whole task
# usage: python main.py -g use_gpu -e epochs -b batch_size -lr learning_rate -wd weight_decay

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from args import process_command
from data import RFMDataset
from model import LogisticReg

# hyperparemeters
arguments = process_command()
epochs        = arguments.epoch
batch_size    = arguments.batch
learning_rate = arguments.lr
weight_decay  = arguments.wd
use_gpu = torch.cuda.is_available()


if __name__ == '__main__':
    # read preprocessed data
    print( 'preparing data...' )
    with open( './data/preprocess-RFM.pickle', 'rb' ) as f:
        X, y, test_X, test_y, idx_to_label_dict, header = pickle.load( f )
        f.close()

    cut = int( len( X ) * 0.1 )
    train_X, train_y = torch.Tensor( X[:cut] ), torch.Tensor( y[:cut] )
    val_X, val_y      = torch.Tensor( X[cut:] ), torch.Tensor( y[cut:] )
    test_X, test_y   = torch.Tensor( test_X ), torch.Tensor( test_y )
    
    train_dataset = RFMDataset( train_X, train_y )
    train_loader  = DataLoader( train_dataset, batch_size=batch_size, shuffle=True )
    
    val_dataset = RFMDataset( val_X, val_y )
    val_loader  = DataLoader( val_dataset, batch_size=batch_size, shuffle=False )
    
    test_dataset = RFMDataset( test_X, test_y )
    test_loader  = DataLoader( test_dataset, batch_size=batch_size, shuffle=False )

    print( 'training initializing...' )
    model = LogisticReg( len( X[0] ), len( set( y ) ) ) # define model
    critirion = nn.CrossEntropyLoss() # define loss function
    optimizer = optim.Adam( model.parameters(), lr=learning_rate, weight_decay=weight_decay ) # define optimizer
    
    if use_gpu:
        model.cuda()


    train_epoch_loss = []
    train_epoch_acc  = []
    val_epoch_loss   = []
    val_epoch_acc    = []

    print( 'start training...' )
    for epoch in tqdm( range( epochs ) ):
        
        # training mode
        model.train()

        # record data in an epoch
        train_loss = []
        train_acc  = []

        # this way is suggested by some god in pytorch forum
        for idx, ( X_, y_ ) in enumerate( train_loader ):
            if use_gpu:
                X_ = X_.cuda() # put X to gpu
                y_ = y_.cuda()
            
            optimizer.zero_grad() # because PyTorch accumulates the gradients on subsequent backward passes
            output = model( X_ ) # predict on training
            loss = critirion( output, y_.long() ) # long is like one-hot encoding
            loss.backward() # compute gradient
            optimizer.step() # update paras
            
            predict = torch.max( output, 1 )[1] 
            acc = np.mean( ( y_ == predict ).cpu().numpy() )

            train_acc.append( acc )
            train_loss.append( loss.item() )
        
        # record epoch training loss    
        train_epoch_loss.append( (epoch + 1, np.mean(train_loss)) )
        train_epoch_acc.append( (epoch + 1, np.mean(train_acc)) )

        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))

        # testing mode, if there is dropout or sth. it will be different from training mode
        model.eval()

        with torch.no_grad(): # close autograd engine to speed up computation
            valid_loss = []
            valid_acc  = []

            for idx, ( X_, y_ ) in enumerate( val_loader ):
                if use_gpu:
                    X_ = X_.cuda()
                    y_ = y_.cuda()
                    
                output = model( X_ )
                loss = critirion( output, y_.long() )
                predict = torch.max( output, 1 )[1]

                acc = np.mean( ( y_ == predict ).cpu().numpy() )
                valid_loss.append( loss.item() )
                valid_acc.append( acc )
                 
            val_epoch_acc.append( ( epoch + 1, np.mean( np.mean( valid_acc ) ) ) )
            val_epoch_loss.append( ( epoch + 1, np.mean( np.mean( valid_loss ) ) ) )
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))


    with open( './data/history.pickle', 'wb' ) as f:
        pickle.dump( ( (val_epoch_loss, train_epoch_loss), ( val_epoch_acc, train_epoch_loss ) ), f )
        f.close()
        
#     model.eval()
#     with torch.no_grad():
#         test_acc = []
        
#         for idx, ( X_, y_ ) in enumerate( test_loader ):
#             if use_gpu:
#                 X_ = X_.cuda()
#                 y_ = y_.cuda()
                
#             output = model( X_ )
#             loss = critirion( output, y_.long() )
#             predict = torch.max( output, 1 )[1]
            
#             acc = np.mean( ( y_ == predict ).cpu().numpy() )
#             test_acc.append( acc )
            
#         print("Epoch: {}, test acc: {:.4f}".format(epoch + 1, np.mean(test_acc)))
