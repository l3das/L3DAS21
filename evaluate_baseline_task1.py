import sys, os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as utils
from metrics import task1_metric
from FaSNet import FaSNet_origin
from utility_functions import load_model, save_model

'''
Load pretrained FasNet model and compute the metric for
the Task 1 of the L3DAS21 challenge.
The metric is: (STOI+(1-WER))/2
'''
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

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    #LOAD MODEL
    model = FaSNet_origin(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                          hidden_dim=args.hidden_dim, layer=args.layer,
                          segment_size=args.segment_size, nspk=args.nspk,
                          win_len=args.win_len, context_len=args.context_len,
                          sr=args.sr)
    if args.use_cuda:
        print("move model to gpu")
    model = model.to(device)

    #load checkpoint
    state = load_model(model, None, args.model_path, args.use_cuda)

    #COMPUTING METRICS
    print("COMPUTING METRICS")
    WER = 0.
    STOI = 0.
    METRIC = 0.
    count = 0
    model.eval()
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)

            outputs = model(x, torch.tensor([0.]))

            outputs = outputs[:,0,:].cpu().numpy()
            target = target.cpu().numpy()
            print ('SSSSSS',outputs.shape, target.shape)

            outputs = np.squeeze(outputs)
            target = np.squeeze(target)

            #outputs = outputs / np.max(outputs) * 0.9  #normalize prediction
            metric, wer, stoi = task1_metric(target, outputs)

            METRIC += (1. / float(example_num + 1)) * (metric - METRIC)
            WER += (1. / float(example_num + 1)) * (wer - WER)
            STOI += (1. / float(example_num + 1)) * (stoi - STOI)
            '''
            #save sounds
            if count % 50 == 0:
                sf.write(os.path.join(args.results_path, str(example_num)+'.wav'), outputs, 16000, 'PCM_16')
            '''
            print ('metric: ', metric, 'wer: ', wer, 'stoi: ', stoi)

            pbar.update(1)


    #visualize and save results
    results = {'word error rate': WER,
               'stoi': STOI,
               'task 1 metric': METRIC
               }

    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(args.results_path, 'task1_metrics_dict.json')
    np.save(out_path, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/fasnet_fulltrain100_REAL/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/fasnet_fulltrain100_REAL/metrics')
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task1_mini/task1_predictors_test.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task1_mini/task1_target_test.pkl')
    parser.add_argument('--sr', type=int, default=16000)
    #model parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--segment_size', type=int, default=24)
    parser.add_argument('--nspk', type=int, default=2)
    parser.add_argument('--win_len', type=int, default=16)
    parser.add_argument('--context_len', type=int, default=16)

    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)

    main(args)
