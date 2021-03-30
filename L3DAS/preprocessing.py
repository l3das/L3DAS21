import argparse
'''
Take as input the downloaded dataset (audio files and target data)
and output pytorch datasets for task1 and task2, separately.
Command line inputs define which task to process and its parameters
'''


def preprocessing_task1():
    sr_task1 = 16000
    train100_folder = 'train'
    train360_folder = 'train360'
    dev_folder = 'dev'

    sets = [train100_folder, dev_folder]

    for main_folder in sets:
        contents = os.listdir(main_folder)
        print (contents)

    #create pytorch dataset with the preprocessed data
    #seve it to args.output_directory

def preprocessing_task2():
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
                        
    parser.add_argument('--stft_nparseg', type=int, default=256,
                        help='num of stft frames')
    parser.add_argument('--stft_noverlap', type=int, default=128,
                        help='num of overlapping samples for stft')
    parser.add_argument('--stft_noverlap', type=str, default='hamming',
                        help='stft window_type')

    args = parser.parse_args()


    if args.task_number == 1:
        preprocessing_task1(args)
    elif args.task_number == 2:
        preprocessing_task2(args)
