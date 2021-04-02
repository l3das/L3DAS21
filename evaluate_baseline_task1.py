import sys, os
import time
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
from waveunet_model.waveunet import Waveunet
import waveunet_model.utils as model_utils
from metrics import task1_metric

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dyn_pad(x, y, size_x=169641, size_y=160089):

    pad_x = torch.zeros(x.shape[0],x.shape[1], size_x)
    pad_y = torch.zeros(y.shape[0],y.shape[1], size_y)
    pad_x[:,:,:x.shape[-1]] = x
    pad_y[:,:,:y.shape[-1]] = y
    return pad_x, pad_y

def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    print ('\nLoading dataset')
    #LOAD DATASET
    with open(args.predictors_path, 'rb') as f:
        predictors = pickle.load(f)
    with open(args.target_path, 'rb') as f:
        target = pickle.load(f)
    predictors = np.array(predictors)
    target = np.array(target)

    print ('\nShapes:')
    print ('Predictors: ', predictors.shape)

    #convert to tensor
    predictors = torch.tensor(predictors).float()
    target = torch.tensor(target).float()
    #build dataset from tensors
    dataset_ = utils.TensorDataset(predictors, target)
    #build data loader from dataset
    dataloader = utils.DataLoader(dataset_, 1, shuffle=False, pin_memory=True)

    #LOAD MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=160000, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)
    if args.use_cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
    model = model.to(device)
    state = model_utils.load_model(model, None, args.model_path, args.use_cuda)

    #COMPUTING METRICS
    print("COMPUTING METRICS")
    model.eval()
    WER = 0.
    STOI = 0.
    METRIC = 0.
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            x, target = dyn_pad(x, target)
            target = target.to(device)
            x = x.to(device)

            outputs = model(x, 'vocals')
            outputs = outputs['vocals'].cpu().numpy()
            target = target.cpu().numpy()
            outputs = np.squeeze(outputs)
            target = np.squeeze(target)

            metric, wer, stoi = task1_metric(target, outputs)
            noise = np.random.sample(len(target)) * 2 - 1
            #metric, wer, stoi = task1_metric(target, noise)

            metric += (1. / float(example_num + 1)) * (metric - METRIC)
            wer += (1. / float(example_num + 1)) * (wer - WER)
            stoi += (1. / float(example_num + 1)) * (stoi - STOI)
            sf.write(os.path.join(args.results_path, str(example_num)+'.wav'), outputs, 16000, 'PCM_16')
            #librosa.output.write_wav(os.path.join(args.results_path, str(example_num)+'.wav'), outputs, 16000)

            print (metric, wer, stoi)

        #$test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.update(1)



    results = {'word error rate': WER,
               'stoi': STOI,
               'task 1 metric': METRIC
               }

    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(args.results_path, 'task1_metrics_dict.json')
    np.save(out_path, results)
    #writer.add_scalar("test_loss", test_loss, state["step"])




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/waveunet_TRAINED/checkpoints/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/waveunet_TRAINED')

    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task1/task1_predictors_test.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task1/task1_target_test.pkl')
    #model parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--instruments', type=str, nargs='+', default=["vocals"],
                        help="List of instruments to separate (default: \"vocals\")")
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=4,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)

    main(args)
