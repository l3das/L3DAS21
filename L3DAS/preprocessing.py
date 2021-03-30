import argparse
import os
import numpy as np
import librosa
import pickle
'''
Take as input the downloaded dataset (audio files and target data)
and output pytorch datasets for task1 and task2, separately.
Command line inputs define which task to process and its parameters
'''


def preprocessing_task1(args):
    sr_task1 = 16000
    train100_folder = 'train'
    train360_folder = 'train360'
    dev_folder = 'test'

    sets = [train100_folder, dev_folder]
    predictors_train = []
    predictors_validation = []
    predictors_test = []
    target_train = []
    target_validation = []
    target_test = []
    count = 0
    for folder in sets:
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
                    if args.num_mics == 2:  # if both ambisonics mics are wanted
                        #stack the additional 4 channels to get a (8, samples) shape
                        B_sound_path = sound_path.replace('A', 'B')
                        samples_B, sr = librosa.load(sound_path, sr_task1, mono=False)
                        samples = np.vstack((samples,samples_B))
                    samples_target, sr = librosa.load(target_path, sr_task1, mono=False)
                    #append to final arrays
                    if folder == dev_folder:
                        predictors_test.append(samples)
                        target_test.append(samples_target)
                    else:
                        predictors_train.append(samples)
                        target_train.append(samples_target)
                    count += 1
                    if args.num_data != 0 and count >= args.num_data and count >= args.num_data:
                        break

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_train = predictors_train[:split_point]
    target_train = target_train[:split_point]
    predictors_validation = predictors_validation[split_point:]
    target_validation = target_validation[split_point:]


    with open('predictors_train.npy', 'wb') as f:
        pickle.dump(predictors_train, f)
    with open('predictors_validation.npy', 'wb') as f:
        pickle.dump(predictors_validation, f)
    with open('predictors_test.npy', 'wb') as f:
        pickle.dump(predictors_test, f)
    with open('target_train.npy', 'wb') as f:
        pickle.dump(target_train, f)
    with open('target_validation.npy', 'wb') as f:
        pickle.dump(target_validation, f)
    with open('target_test.npy', 'wb') as f:
        pickle.dump(target_test, f)

    #create pytorch dataset with the preprocessed data
    #seve it to args.output_directory

def preprocessing_task2(args):
    sr_task2 = 32000
    #create pytorch dataset with the preprocessed data
    #seve it to args.output_directory


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    #i/o
    parser.add_argument('--task_number', type=int,
                        help='task to be pre-processed')
    parser.add_argument('--input_path', type=str,
                        help='directory where the dataset has been downloaded')
    parser.add_argument('--output_path', type=str, default='processed',
                        help='where to save the numpy matrices')

    #processing type
    parser.add_argument('--processsing_type', type=str, default='stft',
                        help='stft or waveform')
    parser.add_argument('--train_val_split', type=float, default=0.7,
                        help='perc split between train and validation sets')
    parser.add_argument('--num_mics', type=int, default=1,
                        help='how many ambisonics mics (1 or 2)')
    parser.add_argument('--num_data', type=float, default=100,
                        help='how many datapoints per set. 0 means all available data')
    parser.add_argument('--stft_nparseg', type=int, default=256,
                        help='num of stft frames')
    parser.add_argument('--stft_noverlap', type=int, default=128,
                        help='num of overlapping samples for stft')
    parser.add_argument('--stft_window', type=str, default='hamming',
                        help='stft window_type')

    args = parser.parse_args()


    if args.task_number == 1:
        preprocessing_task1(args)
    elif args.task_number == 2:
        preprocessing_task2(args)
