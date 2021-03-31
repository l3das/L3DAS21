import sys, os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as utils
from waveunet_model.waveunet import Waveunet


parser = argparse.ArgumentParser()
#saving parameters
parser.add_argument('--results_folder', type=str, default='../results')
parser.add_argument('--results_path', type=str, default='../results/results_task1.npy')
parser.add_argument('--model_path', type=str, default='../results/model_task1')
#dataset parameters
parser.add_argument('--training_predictors_path', type=str, default='../prova_pickle/training_predictors.pkl')
parser.add_argument('--training_target_path', type=str, default='../prova_pickle/training_target.pkl')
parser.add_argument('--validation_predictors_path', type=str, default='../prova_pickle/validation_predictors.pkl')
parser.add_argument('--validation_target_path', type=str, default='../prova_pickle/validation_target.pkl')
parser.add_argument('--test_predictors_path', type=str, default='../prova_pickle/test_predictors.pkl')
parser.add_argument('--test_target_path', type=str, default='../prova_pickle/test_target.pkl')
#training parameters
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--use_cuda', type=str, default='False')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.00005)
parser.add_argument('--regularization_lambda', type=float, default=0.)
parser.add_argument('--early_stopping', type=str, default='True')
parser.add_argument('--save_model_metric', type=str, default='total_loss')
parser.add_argument('--load_pretrained', type=str, default=None)
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_fold', type=int, default=1)
parser.add_argument('--fixed_seed', type=str, default='True')
#model parameters
parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                    help="List of instruments to separate (default: \"bass drums other vocals\")")
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of data loader worker threads (default: 1)')
parser.add_argument('--features', type=int, default=32,
                    help='Number of feature channels per layer')
parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                    help='Folder to write logs into')
parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                    help='Dataset path')
parser.add_argument('--hdf_dir', type=str, default="hdf",
                    help='Dataset path')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                    help='Folder to write checkpoints into')
parser.add_argument('--load_model', type=str, default=None,
                    help='Reload a previously trained model (whole task model)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate in LR cycle (default: 1e-3)')
parser.add_argument('--min_lr', type=float, default=5e-5,
                    help='Minimum learning rate in LR cycle (default: 5e-5)')
parser.add_argument('--cycles', type=int, default=2,
                    help='Number of LR cycles per epoch')
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch size")
parser.add_argument('--levels', type=int, default=6,
                    help="Number of DS/US blocks")
parser.add_argument('--depth', type=int, default=1,
                    help="Number of convs per block")
parser.add_argument('--sr', type=int, default=44100,
                    help="Sampling rate")
parser.add_argument('--channels', type=int, default=2,
                    help="Number of input audio channels")
parser.add_argument('--kernel_size', type=int, default=5,
                    help="Filter width of kernels. Has to be an odd number")
parser.add_argument('--output_size', type=float, default=2.0,
                    help="Output duration")
parser.add_argument('--strides', type=int, default=4,
                    help="Strides in Waveunet")
parser.add_argument('--patience', type=int, default=20,
                    help="Patience for early stopping on validation set")
parser.add_argument('--example_freq', type=int, default=200,
                    help="Write an audio summary into Tensorboard logs every X training iterations")
parser.add_argument('--loss', type=str, default="L1",
                    help="L1 or L2")
parser.add_argument('--conv_type', type=str, default="gn",
                    help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
parser.add_argument('--res', type=str, default="fixed",
                    help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
parser.add_argument('--separate', type=int, default=1,
                    help="Train separate model for each source (1) or only one (0)")
parser.add_argument('--feature_growth', type=str, default="double",
                    help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

#PUT PARAMETERS FOR WAVE U NET


args = parser.parse_args()

#eval string args
args.use_cuda = eval(args.use_cuda)
args.early_stopping = eval(args.early_stopping)
args.fixed_seed = eval(args.fixed_seed)

if args.use_cuda:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'

if args.fixed_seed:
    seed = 1
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


print ('\nLoading dataset')


#load dataset

with open(args.training_predictors_path, 'rb') as f:
    training_predictors = pickle.load(f)
with open(args.training_target_path, 'rb') as f:
    training_target = pickle.load(f)
with open(args.validation_predictors_path, 'rb') as f:
    validation_predictors = pickle.load(f)
with open(args.validation_target_path, 'rb') as f:
    validation_target = pickle.load(f)
with open(args.test_predictors_path, 'rb') as f:
    test_predictors = pickle.load(f)
with open(args.test_target_path, 'rb') as f:
    test_target = pickle.load(f)
'''
training_predictors = np.load(args.training_predictors_path, allow_pickle=True)
training_target = np.load(args.training_target_path, allow_pickle=True)
validation_predictors = np.load(args.validation_predictors_path, allow_pickle=True)
validation_target = np.load(args.validation_target_path, allow_pickle=True)
test_predictors = np.load(args.test_predictors_path, allow_pickle=True)
test_target = np.load(args.test_target_path, allow_pickle=True)

'''
training_predictors = np.array(training_predictors)
training_target = np.array(training_target)
validation_predictors = np.array(validation_predictors)
validation_target = np.array(validation_target)
test_predictors = np.array(test_predictors)
test_target = np.array(test_target)
'''
#reshaping
training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])
'''

print ('\nShapes:')
print ('Training predictors: ', training_predictors.shape)
print ('Validation predictors: ', validation_predictors.shape)
print ('Test predictors: ', test_predictors.shape)



#convert to tensor
training_predictors = torch.tensor(training_predictors).float()
validation_predictors = torch.tensor(validation_predictors).float()
test_predictors = torch.tensor(test_predictors).float()
training_target = torch.tensor(training_target).float()
validation_target = torch.tensor(validation_target).float()
test_target = torch.tensor(test_target).float()
#build dataset from tensors
tr_dataset = utils.TensorDataset(training_predictors, training_target)
val_dataset = utils.TensorDataset(validation_predictors, validation_target)
test_dataset = utils.TensorDataset(test_predictors, test_target)
#build data loader from dataset
tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)  #no batch here!!



#LOAD MODEL
num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
               [args.features*2**i for i in range(0, args.levels)]
target_outputs = int(args.output_size * args.sr)
model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                 target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                 conv_type=args.conv_type, res=args.res, separate=args.separate)

if args.use_cuda:
    model = model_utils.DataParallel(model)
    print("move model to gpu")
model = model.to(device)
sys.exit(0)


#print (model)

#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

#define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.regularization_lambda)
loss_function = locals()[args.loss_function]

#init history
train_loss_hist = []
val_loss_hist = []

loading_time = float(time.perf_counter()) - float(loading_start)
print ('\nLoading time: ' + str(np.round(float(loading_time), decimals=1)) + ' seconds')


criterion = nn.MSELoss()

def train_model():
    for epoch in range(args.num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        print ('\n')
        string = 'Epoch: [' + str(epoch+1) + '/' + str(args.num_epochs) + '] '
        #iterate batches
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, v, a, d = model(sounds)
            loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)

            l = criterion(sounds,recon)
            l.backward()
            #loss['total'].backward(retain_graph=True)
            #lotal_loss.backward()
            #print ('criterion:')
            optimizer.step()
            #loss['total'] = loss['total'].detach()
            #print progress
            perc = int(i / len(tr_data) * 20)
            inv_perc = int(20 - perc - 1)
            loss_print_t = str(np.round(loss['total'].detach().item(), decimals=5))
            #loss_print_t = str(np.round(loss.detach().item(), decimals=5))

            string_progress = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t
            print ('\r', string_progress, end='')
            #del loss

        #create history
        train_batch_losses = []
        val_batch_losses = []
        with torch.no_grad():
            model.eval()
            #training data
            for i, (sounds, truth) in enumerate(tr_data):
                sounds = sounds.to(device)
                truth = truth.to(device)

                recon, v, a, d = model(sounds)
                loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)

                train_batch_losses.append(loss)
            #validation data
            for i, (sounds, truth) in enumerate(val_data):
                sounds = sounds.to(device)
                truth = truth.to(device)

                recon, v, a, d = model(sounds)
                loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)

                val_batch_losses.append(loss)
        #append to history and print
        train_epoch_loss = {'total':[], 'emo':[], 'recon':[],
                            'valence':[],'arousal':[], 'dominance':[]}
        val_epoch_loss = {'total':[], 'emo':[], 'recon':[],
                          'valence':[],'arousal':[], 'dominance':[]}

        for i in train_batch_losses:
            for j in i:
                name = j
                value = i[j]
                train_epoch_loss[name].append(value.item())
        for i in val_batch_losses:
            for j in i:
                name = j
                value = i[j]
                val_epoch_loss[name].append(value.item())

        for i in train_epoch_loss:
            train_epoch_loss[i] = np.mean(train_epoch_loss[i])
            val_epoch_loss[i] = np.mean(val_epoch_loss[i])
        print ('\n EPOCH LOSSES:')
        print ('\n Training:')
        print (train_epoch_loss)
        print ('\n Validation:')
        print (val_epoch_loss)


        train_loss_hist.append(train_epoch_loss)
        val_loss_hist.append(val_epoch_loss)

        #print ('\n  Train loss: ' + str(np.round(train_epoch_loss.item(), decimals=5)) + ' | Val loss: ' + str(np.round(val_epoch_loss.item(), decimals=5)))

        #compute epoch time
        epoch_time = float(time.perf_counter()) - float(epoch_start)
        print ('\n Epoch time: ' + str(np.round(float(epoch_time), decimals=1)) + ' seconds')

        #save best model (metrics = validation loss)
        if epoch == 0:
            torch.save(model.state_dict(), args.model_path)
            print ('\nModel saved')
            saved_epoch = epoch + 1
        else:
            if args.save_model_metric == 'total_loss':
                best_loss = min([i['total'] for i in val_loss_hist[:-1]])
                #best_loss = min(val_loss_hist['total'].item()[:-1])  #not looking at curr_loss
                curr_loss = val_loss_hist[-1]['total']
                if curr_loss < best_loss:
                    torch.save(model.state_dict(), args.model_path)
                    print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                    saved_epoch = epoch + 1


            else:
                raise ValueError('Wrong metric selected')
        '''
        if args.num_experiment != 0:
            #print info on dataset, experiment and instance if performing a grid search
            utilstring = 'dataset: ' + str(args.dataset) + ', exp: ' + str(args.num_experiment) + ', run: ' + str(args.num_run) + ', fold: ' + str(args.num_fold)
            print ('')
            print (utilstring)
        '''

        if args.early_stopping and epoch >= args.patience+1:
            patience_vec = [i['total'] for i in val_loss_hist[-args.patience+1:]]
            #patience_vec = val_loss_hist[-args.patience+1:]
            best_l = np.argmin(patience_vec)
            if best_l == 0:
                print ('Training early-stopped')
                break


    #COMPUTE
    model.load_state_dict(torch.load(args.model_path), strict=False)  #load best model
    train_batch_losses = []
    val_batch_lesses = []
    test_batch_losses = []

    model.eval()
    with torch.no_grad():
        #TRAINING DATA
        for i, (sounds, truth) in enumerate(tr_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, v, a, d = model(sounds)
            loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)

            train_batch_losses.append(loss)

        #VALIDATION DATA
        for i, (sounds, truth) in enumerate(val_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, v, a, d = model(sounds)
            loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)

            val_batch_losses.append(loss)

        #TEST DATA
        for i, (sounds, truth) in enumerate(test_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, v, a, d = model(sounds)
            loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)

            test_batch_losses.append(loss)

    #compute final mean of batch losses for train, validation and test set
    train_loss = {'total':[], 'emo':[], 'recon':[],
                  'valence':[], 'arousal':[], 'dominance':[]}
    val_loss = {'total':[], 'emo':[], 'recon':[],
                'valence':[],'arousal':[], 'dominance':[]}
    test_loss = {'total':[], 'emo':[], 'recon':[],
                 'valence':[],'arousal':[], 'dominance':[]}
    for i in train_batch_losses:
        for j in i:
            name = j
            value = i[j]
            train_loss[name].append(value.item())
    for i in val_batch_losses:
        for j in i:
            name = j
            value = i[j]
            val_loss[name].append(value.item())
    for i in test_batch_losses:
        for j in i:
            name = j
            value = i[j]
            test_loss[name].append(value.item())

    for i in train_loss:
        train_loss[i] = np.mean(train_loss[i])
        val_loss[i] = np.mean(val_loss[i])
        test_loss[i] = np.mean(test_loss[i])


    #save results in temp dict file
    temp_results = {}

    #save loss
    temp_results['train_loss_total'] = train_loss['total']
    temp_results['val_loss_total'] = val_loss['total']
    temp_results['test_loss_total'] = test_loss['total']

    temp_results['train_loss_recon'] = train_loss['recon']
    temp_results['val_loss_recon'] = val_loss['recon']
    temp_results['test_loss_recon'] = test_loss['recon']

    temp_results['train_loss_emo'] = train_loss['emo']
    temp_results['val_loss_emo'] = val_loss['emo']
    temp_results['test_loss_emo'] = test_loss['emo']

    temp_results['train_loss_valence'] = train_loss['valence']
    temp_results['val_loss_valence'] = val_loss['valence']
    temp_results['test_loss_valence'] = test_loss['valence']

    temp_results['train_loss_arousal'] = train_loss['arousal']
    temp_results['val_loss_arousal'] = val_loss['arousal']
    temp_results['test_loss_arousal'] = test_loss['arousal']

    temp_results['train_loss_dominance'] = train_loss['dominance']
    temp_results['val_loss_dominance'] = val_loss['dominance']
    temp_results['test_loss_dominance'] = test_loss['dominance']

    temp_results['train_loss_hist'] = train_loss_hist
    temp_results['val_loss_hist'] = train_loss_hist
    temp_results['parameters'] = vars(args)


    np.save(args.results_path, temp_results)

    #print  results
    print ('\nRESULTS:')
    keys = list(temp_results.keys())
    keys.remove('parameters')
    keys.remove('train_loss_hist')
    keys.remove('val_loss_hist')

    train_keys = [i for i in keys if 'train' in i]
    val_keys = [i for i in keys if 'val' in i]
    test_keys = [i for i in keys if 'test' in i]


    print ('\n train:')
    for i in train_keys:
        print (i, ': ', temp_results[i])
    print ('\n val:')
    for i in val_keys:
        print (i, ': ', temp_results[i])
    print ('\n test:')
    for i in test_keys:
        print (i, ': ', temp_results[i])

train_model()
