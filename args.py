# this script is for defining argparser, which makes training model lazier

import argparse

def process_command():
    
    parser = argparse.ArgumentParser( description='Hyperparameters' )
    parser.add_argument( '--gpu', '-g', default=True, help='use gpu or not', type=bool, dest='gpu' )
    # parser.add_argument('--model', '-model', default='./model', help='path of model' )
    parser.add_argument( '--epoch', '-e', default=300, help='# of epochs', type=int, dest='epoch' )
    parser.add_argument( '--batch', '-b', default=64, help='batch size', type=int, dest='batch' )
    parser.add_argument( '--learing_rate', '-lr', default=1e-4, help='learing rate', type=float, dest='lr' )
    parser.add_argument( '--weight_decay', '-wd', default=0.0, help='l2-penalty for torch', type=float, dest='wd' )

    return parser.parse_args()
