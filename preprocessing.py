import argparse
import os, sys
import numpy as np
import librosa
import pickle
import random
from utility_functions import get_label_task2
'''
Take as input the unzipped dataset folders and output pickle lists
containing the pre-processed data for task1 and task2, separately.
Separate training, validation and test matrices are saved.
Command line inputs define which task to process and its parameters.
'''

def preprocessing_task1(args):
    sr_task1 = 16000

    def pad(x, size=sr_task1*10):
        #pad all sounds to 10 seconds
        length = x.shape[-1]
        if length > size:
            pad = x[:,:size]
        else:
            pad = np.zeros((x.shape[0], size))
            pad[:,:length] = x
        return pad

    def process_folder(folder, args):
        print ('Processing ' + folder + ' folder...')
        predictors = []
        target = []
        count = 0
        main_folder = os.path.join(args.input_path, folder)
        contents = os.listdir(main_folder)
        for sub in contents:
            sub_folder = os.path.join(main_folder, sub)
            contents_sub = os.listdir(sub_folder)
            for lower in contents_sub:
                lower_folder = os.path.join(sub_folder, lower)
                data_path = os.path.join(lower_folder, 'data')
                data = os.listdir(data_path)
                data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
                for sound in data:
                    sound_path = os.path.join(data_path, sound)
                    target_path = sound_path.replace('data', 'labels').replace('_A', '')
                    samples, sr = librosa.load(sound_path, sr_task1, mono=False)
                    samples = pad(samples)
                    if args.num_mics == 2:  # if both ambisonics mics are wanted
                        #stack the additional 4 channels to get a (8, samples) shape
                        B_sound_path = sound_path.replace('A', 'B')
                        samples_B, sr = librosa.load(B_sound_path, sr_task1, mono=False)
                        samples_B = pad(samples_B)
                        samples = np.vstack((samples,samples_B))
                    samples_target, sr = librosa.load(target_path, sr_task1, mono=False)
                    samples_target = samples_target.reshape((1, samples_target.shape[0]))
                    samples_target = pad(samples_target)
                    #append to final arrays

                    if args.segmentation_len in not None:
                        #segment longer file to shorter frames
                        segmentation_len_samps = int(sr_task1 * args.segmentation_len)
                        predictors_cuts, target_cuts = uf.segment_waveforms(samples, samples_target, segmentation_len_samps)
                        for i in range(len(predictors_cuts)):
                            predictors.append(predictors_cuts[i])
                            taget.append(target_cut[i])
                            print (predictors_cuts[i].shape, target_cut[i].shape)
                    else:
                        predictors.append(samples)
                        target.append(samples_target)
                    count += 1
                    if args.num_data is not None and count >= args.num_data:
                        break
                else:
                    continue
                break
            else:
                continue
            break

        return predictors, target

    predictors_test, target_test = process_folder('L3DAS_Task1_dev', args)
    if args.training_set == 'train100':
        predictors_train, target_train = process_folder('L3DAS_Task1_train100', args)
    elif args.training_set == 'train360':
        predictors_train, target_train = process_folder('L3DAS_Task1_train360', args)
    elif args.training_set == 'both':
        predictors_train100, target_train100 = process_folder('L3DAS_Task1_train100')
        predictors_train360, target_train360 = process_folder('L3DAS_Task1_train360')
        predictors_train = predictors_train100 + predictors_train360
        target_train = target_train100 + target_train360

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_training = predictors_train[:split_point]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = predictors_train[split_point:]
    target_validation = target_train[split_point:]

    print ('Saving files')
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path,'task1_predictors_train.pkl'), 'wb') as f:
        pickle.dump(predictors_training, f)
    with open(os.path.join(args.output_path,'task1_predictors_validation.pkl'), 'wb') as f:
        pickle.dump(predictors_validation, f)
    with open(os.path.join(args.output_path,'task1_predictors_test.pkl'), 'wb') as f:
        pickle.dump(predictors_test, f)
    with open(os.path.join(args.output_path,'task1_target_train.pkl'), 'wb') as f:
        pickle.dump(target_training, f)
    with open(os.path.join(args.output_path,'task1_target_validation.pkl'), 'wb') as f:
        pickle.dump(target_validation, f)
    with open(os.path.join(args.output_path,'task1_target_test.pkl'), 'wb') as f:
        pickle.dump(target_test, f)


def preprocessing_task2(args):
    sr_task2 = 32000
    sound_classes=['Chink_and_clink','Computer_keyboard','Cupboard_open_or_close',
             'Drawer_open_or_close','Female_speech_and_woman_speaking',
             'Finger_snapping','Keys_jangling','Knock',
             'Laughter','Male_speech_and_man_speaking',
             'Printer','Scissors','Telephone','Writing']
    file_size=60.0

    def process_folder(folder, args):
        print ('Processing ' + folder + ' folder...')
        predictors = []
        target = []
        data_path = os.path.join(folder, 'data')
        labels_path = os.path.join(folder, 'labels')

        data = os.listdir(data_path)
        data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']
        count = 0
        for sound in data:
            target_name = 'label_' + sound.replace('_A', '').replace('.wav', '.csv')
            sound_path = os.path.join(data_path, sound)
            target_path = os.path.join(data_path, target_name)
            target_path = target_path.replace('data', 'labels')
            samples, sr = librosa.load(sound_path, sr_task2, mono=False)
            if args.num_mics == 2:  # if both ambisonics mics are wanted
                #stack the additional 4 channels to get a (8, samples) shape
                B_sound_path = sound_path.replace('A', 'B')
                samples_B, sr = librosa.load(B_sound_path, sr_task2, mono=False)
                samples = np.vstack((samples,samples_B))
            predictors.append(samples)

            label = get_label_task2(target_path,0.1,file_size,sr_task2,
                                    sound_classes,int(file_size/(args.frame_len/1000.)))

            target.append(label)
            count += 1
            if args.num_data is not None and count >= args.num_data:
                break


        return predictors, target

    train_folder = os.path.join(args.input_path, 'L3DAS_Task2_train')
    test_folder = os.path.join(args.input_path, 'L3DAS_Task2_dev')

    predictors_train, target_train = process_folder(train_folder, args)
    predictors_test, target_test = process_folder(test_folder, args)

    predictors_test = np.array(predictors_test)
    target_test = np.array(target_test)
    #print (predictors_test.shape, target_test.shape)

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_training = predictors_train[:split_point]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = predictors_train[split_point:]
    target_validation = target_train[split_point:]

    print ('Saving files')
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path,'task2_predictors_train.pkl'), 'wb') as f:
        pickle.dump(predictors_training, f)
    with open(os.path.join(args.output_path,'task2_predictors_validation.pkl'), 'wb') as f:
        pickle.dump(predictors_validation, f)
    with open(os.path.join(args.output_path,'task2_predictors_test.pkl'), 'wb') as f:
        pickle.dump(predictors_test, f)
    with open(os.path.join(args.output_path,'task2_target_train.pkl'), 'wb') as f:
        pickle.dump(target_training, f)
    with open(os.path.join(args.output_path,'task2_target_validation.pkl'), 'wb') as f:
        pickle.dump(target_validation, f)
    with open(os.path.join(args.output_path,'task2_target_test.pkl'), 'wb') as f:
        pickle.dump(target_test, f)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    #i/o
    parser.add_argument('--task', type=int,
                        help='task to be pre-processed')
    parser.add_argument('--input_path', type=str, default='DATASETS/Task1',
                        help='directory where the dataset has been downloaded')
    parser.add_argument('--output_path', type=str, default='DATASETS/processed',
                        help='where to save the numpy matrices')
    #task1 parameters
    parser.add_argument('--training_set', type=str, default='train100',
                        help='which training set: train100, train360 or both')
    parser.add_argument('--segmentation_len', type=float, default=None,
                        help='length of segmented frames in seconds')
    #task2 parameters
    parser.add_argument('--frame_len', type=str, default=100,
                        help='frame length for SELD evaluation (in msecs)')
    #processing type
    parser.add_argument('--processsing_type', type=str, default='waveform',
                        help='stft or waveform')
    parser.add_argument('--train_val_split', type=float, default=0.7,
                        help='perc split between train and validation sets')
    parser.add_argument('--num_mics', type=int, default=1,
                        help='how many ambisonics mics (1 or 2)')
    parser.add_argument('--num_data', type=int, default=None,
                        help='how many datapoints per set. 0 means all available data')
    parser.add_argument('--stft_nparseg', type=int, default=256,
                        help='num of stft frames')
    parser.add_argument('--stft_noverlap', type=int, default=128,
                        help='num of overlapping samples for stft')
    parser.add_argument('--stft_window', type=str, default='hamming',
                        help='stft window_type')

    args = parser.parse_args()

    if args.task == 1:
        preprocessing_task1(args)
    elif args.task == 2:
        preprocessing_task2(args)
