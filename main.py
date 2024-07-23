import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.process import train, test
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_loader.data_factor import data_provider
import warnings
from indicatorcampute import computeindicator
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset_path', type=str, default='ETTh1.csv',required=True)
parser.add_argument('--window_size', type=int, default=96,required=True)
parser.add_argument('--horizon', type=int, default=96,required=True)
# labellen < window_size
parser.add_argument('--labellen', type=int, default=48,required=True)
parser.add_argument('--node_cnt', type=int, default=7,required=True)
parser.add_argument('--train_length', type=float, default=8)
parser.add_argument('--valid_length', type=float, default=1)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0008330969355101197)
parser.add_argument('--multi_layer', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=425)
parser.add_argument('--decay_rate', type=float, default=0.6067585285906487)
parser.add_argument('--dropout_rate', type=float, default=0.26441167609099003)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2587955599548246)
parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')
parser.add_argument('--c_out', type=int, default=12, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=2, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--use_nnorm', type=bool, default=False)
args = parser.parse_args()
data_file = os.path.join('dataset', args.dataset_path)
pathname = args.dataset_path+'_window_'+str(args.window_size)+'horizon'+str(args.horizon)
result_train_file = os.path.join('output', pathname, 'train')
result_test_file = os.path.join('output', pathname, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
train_set, train_loader = data_provider(args, flag='train')
valid_set, valid_loader = data_provider(args, flag='val')
test_set, test_loader = data_provider(args, flag='test')
torch.manual_seed(46)
if __name__ == '__main__':
    if args.train:
        before_train = datetime.now().timestamp()
        _ = train(args, train_set, train_loader,
                                       valid_set, valid_loader, result_train_file)
        after_train = datetime.now().timestamp()
    if args.evaluate:
        mae, rmse = test(args, test_set, test_loader,result_train_file, result_test_file)
        print('mean_mae ={:5.6f} mean_rmse ={:5.6f}'.format(mae, rmse))

# num_features = pre.shape[1]
# fig, axs = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))
# for i in range(num_features):
#     pre_feature = pre.iloc[:, i]
#     tar_feature = tar.iloc[:, i]
#     axs[i].plot(pre_feature, label='Predicted')
#     axs[i].plot(tar_feature, label='True')
#     axs[i].set_xlabel('Time')
#     axs[i].set_ylabel(f'Feature {i + 1}')
#     axs[i].legend()
#     axs[i].grid(True)
# plt.tight_layout()
# plt.show()

