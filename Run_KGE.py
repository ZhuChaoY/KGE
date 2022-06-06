import os
import argparse
import tensorflow as tf
from Models import *
    

parser = argparse.ArgumentParser(description = 'KGE')

#'TransE', 'TransH', 'ConvKB', 'RotatE'
parser.add_argument('--model', type = str, default = 'TransE',
                    help = 'model name') 
#'FB15k', 'FB15k-237', 'WN18', 'WN18RR', 'Kinship', 'UMLS'
parser.add_argument('--dataset', type = str, default = 'FB15k',
                    help = 'dataset')  
parser.add_argument('--dim', type = int, default = 100,
                    help = 'embedding dim')
parser.add_argument('--margin', type = float, default = None,
                    help = 'margin value for TransX')
parser.add_argument('--n_filter', type = int, default = None,
                    help = 'number of filters for ConvKB')
parser.add_argument('--l2', type = float, default = 5e-4,
                    help = 'l2 penalty coefficient')
parser.add_argument('--l_r', type = float, default = 5e-3, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 5000,
                    help = 'batch size for SGD')
parser.add_argument('--epoches', type = int, default = 5000,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 2,
                    help = 'earlystop steps')
parser.add_argument('--do_train', type = int, default = 1,
                    help = 'whether to train')
parser.add_argument('--do_predict', type = int, default = 1,
                    help = 'whether to predict')
parser.add_argument('--do_evaluate', type = int, default = 0,
                    help = 'whether to evaluate')
parser.add_argument('--gpu', type = str, default = '0',
                    help = 'GPU number')


args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #shielding warning
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu #GPU number
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 

model = eval(args.model + '(args)')
model.run(config) 