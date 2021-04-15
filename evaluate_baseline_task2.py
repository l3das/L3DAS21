import sys, os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.data as utils
from torchvision import models
from metrics import location_sensitive_detection
from SELDNet import Seldnet
from utility_functions import load_model, save_model, gen_submission_list_task2

'''
Load pretrained model and compute the metrics for Task 1
of the L3DAS21 challenge. The metric is: (STOI+(1-WER))/2
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
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
    if args.architecture == 'vgg16':
        features_dim = int(target.shape[-2] * target.shape[-1])
        model = models.vgg16()
        model.features[0] = nn.Conv2d(args.input_channels, 64, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1))
        model.classifier[6] =nn.Linear(in_features=4096,
                                    out_features=features_dim, bias=True)
    if args.architecture == 'vgg13':
        features_dim = int(test_target.shape[-2] * test_target.shape[-1])
        model = models.vgg13()
        model.features[0] = nn.Conv2d(args.input_channels, 64, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1))
        model.classifier[6] =nn.Linear(in_features=4096,
                                    out_features=features_dim, bias=True)

    if args.architecture == 'seldnet':
        model = Seldnet(time_dim=args.time_dim, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    n_cnn_filters=args.n_cnn_filters, verbose=args.verbose)

    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #load checkpoint
    state = load_model(model, None, args.model_path, args.use_cuda)

    #COMPUTING METRICS
    print("COMPUTING TASK 2 METRICS")
    TP = 0
    FP = 0
    FN = 0
    count = 0
    model.eval()
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            x = x.to(device)

            if args.architecture == 'seldnet':
                sed, doa = model(x)
            else:
                x = model(x)
                sed = x[:,:args.num_classes*3]
                doa = x[:,args.num_classes*3:]
            sed = sed.cpu().numpy().squeeze()
            doa = doa.cpu().numpy().squeeze()
            target = target.numpy().squeeze()

            doa = doa * args.max_label_distance  #de-normalize xyz (we used tanh in the model)
            '''
            prediction = np.zeros((sed.shape[0],sed.shape[1]+doa.shape[1]))
            prediction[:,:sed.shape[1]] = sed
            prediction[:,sed.shape[1]:] = doa
            '''
            prediction = gen_submit_list(sed, doa)
            target = gen_submit_list(target[:,:args.num_classes*3], target[:,args.num_classes*3:])

            tp, fp, fn = location_sensitive_detection(prediction, target, args.num_frames,
                                                      args.spatial_threshold, False)

            TP += tp
            FP += fp
            FN += fn

            count += 1
            '''
            if count % args.save_preds_freq == 0:
                pass
                #save preds
            '''
            pbar.update(1)
    #compute total F score
    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)
    F_score = (2 * precision * recall) / (precision + recall + sys.float_info.epsilon)

    print ('*******************************')
    print ('F score: ', F_score)
    print ('Precision: ', precision)
    print ('Recall: ', recall)

    #visualize and save results
    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(args.results_path, 'task2_metrics_dict.json')
    np.save(out_path, results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/Task2_test/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2_test/metrics')
    parser.add_argument('--save_sounds_freq', type=int, default=None)
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task2_predictors_test.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task2_target_test.pkl')
    parser.add_argument('--sr', type=int, default=32000)
    parser.add_argument('--max_label_distance', type=float, default=2,
                         help='max value of target loc labels (since the model learnt to predict normalized loc labels)')
    parser.add_argument('--num_frames', type=int, default=600,
                        help='total number of time frames in the predicted seld matrices. (600 for 1-minute sounds with 100msecs frames)')

    #model parameters
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--architecture', type=str, default='vgg16',
                        help="model's architecture, can be vgg13, vgg16 or seldnet")
    parser.add_argument('--input_channels', type=int, default=8,
                        help="4/8 for 1/2 mics, multiply x2 if using also phase information")
    parser.add_argument('--num_classes', type=int, default=14)
    #the following parameters produce a prediction for each 100-msecs frame
    #everithing as in the original SELDNet implementation, but the time pooling and time dim
    parser.add_argument('--spatial_threshold', type=float, default=0.5,
                        help="location threshold for considering a predicted sound correct")


    parser.add_argument('--time_dim', type=int, default=4800)
    parser.add_argument('--freq_dim', type=int, default=256)
    parser.add_argument('--output_classes', type=int, default=14)
    parser.add_argument('--pool_size', type=str, default='[[8,2],[8,2],[2,2]]')
    parser.add_argument('--pool_time', type=str, default='True')
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--n_rnn', type=int, default=2)
    parser.add_argument('--fc_size', type=int, default=128)
    parser.add_argument('--dropout_perc', type=float, default=0.)
    parser.add_argument('--n_cnn_filters', type=float, default=64)
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--sed_loss_weight', type=float, default=1.)
    parser.add_argument('--doa_loss_weight', type=float, default=50.)


    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)
    args.pool_size= eval(args.pool_size)
    args.pool_time = eval(args.pool_time)
    args.verbose = eval(args.verbose)

    main(args)
