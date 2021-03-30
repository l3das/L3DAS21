import os, sys
import numpy as np


def gen_seld_out(n_frames, n_overlaps=3, n_classes=14):
    '''
    generate a fake output of the seld model
    ***only for testing
    '''
    results = []
    for frame in range(n_frames):
        n_sounds = np.random.randint(4)
        for i in range(n_sounds):
            t_class = np.random.randint(n_classes)
            tx = (np.random.sample() * 4) - 2
            ty = ((np.random.sample() * 2) - 1) * 1.5
            tz = (np.random.sample() * 2) - 1
            temp_entry = [frame, t_class, tx, ty, tz]
            #print (temp_entry)
            results.append(temp_entry)
    results = np.array(results)
    #pd.DataFrame(results).to_csv(out_path, index=None, header=None)
    return results

def gen_dummy_seld_results(out_path, n_frames=10, n_files=30, perc_tp=0.6,
                           n_overlaps=3, n_classes=14):
    '''
    generate a fake pair of seld model output and truth files
    ***only for testing
    '''

    truth_path = os.path.join(out_path, 'truth')
    pred_path = os.path.join(out_path, 'pred')
    if not os.path.exists(truth_path):
        os.makedirs(truth_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    for file in range(n_files):
        #generate rtandom prediction and truth files
        pred_results = gen_seld_out(n_frames, n_overlaps, n_classes)
        truth_results = gen_seld_out(n_frames, n_overlaps, n_classes)

        #change a few entries in the pred in order to make them match
        num_truth = len(truth_results)
        num_pred = len(pred_results)
        num_tp = int(num_truth * perc_tp)
        list_entries = list(range(min(num_truth, num_pred)))
        random.shuffle(list_entries)
        truth_ids = list_entries[:num_tp]
        for t in truth_ids:
            pred_results[t] = truth_results[t]

        truth_out_file = os.path.join(truth_path, str(file) + '.csv')
        pred_out_file = os.path.join(pred_path, str(file) + '.csv')

        pd.DataFrame(truth_results).to_csv(truth_out_file, index=None, header=None)
        pd.DataFrame(pred_results).to_csv(pred_out_file, index=None, header=None)

def gen_dummy_waveforms(n, out_path):
    '''
    Generate random waveforms as example for the submission
    '''
    sr = 16000
    max_len = 10  #secs

    for i in range(n):
        len = int(np.random.sample() * max_len * sr)
        sound = ((np.random.sample(len) * 2) - 1) * 0.9
        filename = os.path.join(out_path, str(i) + '.npy')
        np.save(filename, sound)

def print_bar(index, total):
    perc = int(index / total * 20)
    perc_progress = int(np.round((float(index)/total) * 100))
    inv_perc = int(20 - perc - 1)
    strings = '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' Progress: ' + str(perc_progress) + '%'
    print ('\r', strings, end='')
