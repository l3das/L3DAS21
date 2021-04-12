import sys, os
import time
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
import torch.utils.data as utils
from FaSNet import FaSNet_origin, FaSNet_TAC
from utility_functions import load_model, save_model

'''
Train our baseline model for the Task2 of the L3DAS21 challenge.
This script saves the best model checkpoint, as well as a dict containing
the results (loss and history). To evaluate the performance of the trained model
according to the challenge metrics, please use evaluate_baseline_task2.py.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

def evaluate(model, device, criterion, dataloader):
    #compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)
            outputs = model(x, torch.tensor([0.]))
            loss = criterion(outputs, target)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss

def main(args):
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

    #LOAD DATASET
    print ('\nLoading dataset')

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

    training_predictors = np.array(training_predictors)
    training_target = np.array(training_target)
    validation_predictors = np.array(validation_predictors)
    validation_target = np.array(validation_target)
    test_predictors = np.array(test_predictors)
    test_target = np.array(test_target)

    print ('\nShapes:')
    print ('Training predictors: ', training_predictors.shape)
    print ('Validation predictors: ', validation_predictors.shape)
    print ('Test predictors: ', test_predictors.shape)
    print ('Training target: ', training_target.shape)
    print ('Validation target: ', validation_target.shape)
    print ('Test target: ', test_target.shape)

    features_dim = int(test_target.shape[-2] * test_target.shape[-1])

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
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)

    #LOAD MODEL
    if args.architecture == 'vgg16':
        model = models.vgg16()
        model.features[0] = nn.Conv2d(args.input_channels, 64, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1))
        model.classifier[6] =nn.Linear(in_features=4096,
                                    out_features=features_dim, bias=True)



    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))
    sys.exit(0)
    #set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    #set up optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    #set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    #load model checkpoint if desired
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model, args.use_cuda)

    #TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // args.batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                target = target.to(device)
                x = x.to(device)
                t = time.time()
                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs = model(x, torch.tensor([0.]))
                loss = criterion(outputs, target)
                loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion, val_data)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint")

            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

                # CHECKPOINT
                print("Saving model...")
                save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())

    #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], args.use_cuda)
    #compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion, tr_data)
    val_loss = evaluate(model, device, criterion, val_data)
    test_loss = evaluate(model, device, criterion, test_data)

    #PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print ('RESULTS')
    for i in results:
        if 'hist' not in i:
            print (i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #saving parameters
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2_test',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default='RESULTS/Task2_test',
                        help='Folder to write checkpoints into')
    #dataset parameters
    '''
    parser.add_argument('--training_predictors_path', type=str, default='DATASETS/processed/task2_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default='DATASETS/processed/task2_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='DATASETS/processed/task2_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='DATASETS/processed/task2_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='DATASETS/processed/task2_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='DATASETS/processed/task2_target_test.pkl')
    '''
    parser.add_argument('--training_predictors_path', type=str, default='DATASETS/processed/Task2_mini/task2_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default='DATASETS/processed/Task2_mini/task2_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='DATASETS/processed/Task2_mini/task2_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='DATASETS/processed/Task2_mini/task2_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='DATASETS/processed/Task2_mini/task2_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='DATASETS/processed/Task2_mini/task2_target_test.pkl')

    #training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='False')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=15,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")
    #model parameters
    parser.add_argument('--architecture', type=str, default='vgg16',
                        help="model's architecture")

    parser.add_argument('--input_channels', type=int, default=4,
                        help="4 for 1-mic or 8 for 2-mics configuration")


    args = parser.parse_args()

    #eval string args
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)

    main(args)
