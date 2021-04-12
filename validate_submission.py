import os, sys
import numpy as np
'''
Check if the the submssion folders are valid: all files must have the
correct format, shape and naming.
WORK IN PROGRESS...
'''

def validate_task1_submission(submission_folder, test_folder):
    '''
    Args:
    - submission_folder: folder containing the model's output for task 1 (non zipped).
    - test_folder: folder containing the released test data (non zipped).
    '''
    #this is just a draft

    #read folders and sort them alphabetically
    #contents_submitted = sorted(os.listdir(submission_folder))
    #contents_test = sorted(os.listdir(test_folder))
    contents_submitted = ['a.npy', 'f.npy']
    contents_test = ['a.npy', 'b.npy']

    #check if non.npy files are present
    non_npy = [x for x in contents_submitted if x[-4:] != '.npy']  #non .npy files
    if len(non_npy) > 0:
        raise AssertionError ('Non-.npy files present. Please include only .npy files '
                              'in the submission folder.')

    #check total number of files
    num_files = len(contents_submitted)
    target_num_files = len(contents_test)
    if not num_files == target_num_files:
        raise AssertionError ('Wrong amount of files. Target:' + str(target_num_files) +
                             ', detected:' + str(len(contents_submitted)))

    #check files naming
    if not contents_submitted != contents_test:
        raise AssertionError ('Wrong file naming. Please name each output file '
                               'exactly as its input .waf file, but with .npy extension')

    #check shape file-by-file
    ###TO BE TESTED
    for i in contents_test:
        submitted_path = os.path.join(submission_folder, i)
        test_path = os.path.join(test_folder, i)
        s = np.load(submitted_path, allow_pickle=True)
        t = librosa.load(test_path, sr_task1, mono=False)
        target_shape = t.shape[-1]
        if not s.shape == target_shape:
            raise AssertionError ('Wrong shape for :' + str(i) + '. Target: ' + str(target_shape) +
                                 ', detected:' + str(s.shape))



    print ('Your submission for Task 1 is valid!')


def validate_task2_submission(submission_folder):
    '''
    - Check if the folder contains all files to be submitted for task 2
    - Check if all files have the correct format
    '''
    pass


validate_task1_submission('ccc', 'bbb')
