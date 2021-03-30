import argparse
import os
import numpy as np
import librosa


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
    predictors = []
    target = []
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
                    target_path = sound_path.replace('data', 'target')
                    samples = librosa.load(sound_path, sr_task1, mono=False)
                    if args.num_mics == 2:  # if bot ambisonics mics are wanted
                        B_sound_path = sound_path.replace('A', 'B')
                        samples_B = librosa.load(sound_path, sr_task1, mono=False)
                        samples = np.vstack((samples,samples_B))


                    print (samples.shape)


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
    #processing type
    parser.add_argument('--num_mics', type=int, default=1,
                        help='how many ambisonics mics (1 or 2)')
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
