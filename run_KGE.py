import os
import argparse
import tensorflow.compat.v1 as tf
from Models import *
    
parser = argparse.ArgumentParser(description = 'KGE')

#'TransE', 'TransH', 'TransR', 'TransD', 'ConvKB'
parser.add_argument('--model', type = str, default = 'TransE',
                    help = 'model name') 
parser.add_argument('--dataset', type = str, default = 'WN18',
                    help = 'dataset') #'FB15k', 'FB15k-237', 'WN18', 'WN18RR'
parser.add_argument('--dim', type = int, default = 128,
                    help = 'embedding dimension')
parser.add_argument('--margin', type = float, default = 1.0,
                    help = 'margin value for TransX')
parser.add_argument('--n_filter', type = int, default = None,
                    help = 'number of filters for ConvKB')
parser.add_argument('--dropout', type = float, default = 0.0, 
                    help = 'dropout rate for ConvKB')
parser.add_argument('--l_r', type = float, default = 1e-3, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 4800,
                    help = 'batch size for SGD')
parser.add_argument('--epoches', type = int, default = 500,
                    help = 'training epoches')
parser.add_argument('--do_train', type = bool, default = True,
                    help = 'whether to train')
parser.add_argument('--save_model', type = bool, default = False,
                    help = 'whether to save trained model')
parser.add_argument('--do_predict', type = bool, default = True,
                    help = 'whether to predict')

    
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #shielding warning
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #GPU number
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 

model = eval(args.model + '(args)')
model.run(config) 


