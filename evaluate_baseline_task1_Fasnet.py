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

def dyn_pad(x, y, size_x=32000, size_y=32000):
    '''
    pad_x = torch.zeros(x.shape[0],x.shape[1], size_x)
    pad_y = torch.zeros(y.shape[0],y.shape[1], size_y)
    pad_x[:,:,:x.shape[-1]] = x
    pad_y[:,:,:y.shape[-1]] = y
    '''
    pad_x = x[:,:,:size_x]
    pad_y = y[:,:,:size_y]
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
    #LOAD MODEL

    model = FaSNet_origin(enc_dim=64, feature_dim=64, hidden_dim=128, layer=6, segment_size=24,
                            nspk=2, win_len=16, context_len=16, sr=16000)
    if args.use_cuda:
        print("move model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))
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

            outputs = model(x, torch.tensor([0.]))

            outputs = outputs[:,0,:].cpu().numpy()
            target = target.cpu().numpy()

            outputs = np.squeeze(outputs)
            target = np.squeeze(target)

            outputs = outputs / np.max(outputs) * 0.9
            metric, wer, stoi = task1_metric(target, outputs)
            #noise = np.random.sample(len(target)) * 2 - 1
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
    parser.add_argument('--model_path', type=str, default='RESULTS/fasnet_test/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/fasnet_test')

    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task1/task1_predictors_test.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task1/task1_target_test.pkl')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_mic', type=float, default=4)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='False')

    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size")

    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")

    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")

    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")

    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)

    main(args)
