import os, sys
import numpy as np
import pickle
import math
import pandas as pd
import torch
from scipy.signal import stft
import librosa

'''
Miscellaneous utilities
'''

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
    }, path)


def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state


def spectrum_fast(x, nperseg=512, noverlap=128, window='hamming', cut_dc=True,
                  output_phase=True, cut_last_timeframe=True):
    '''
    Compute magnitude spectra from monophonic signal
    '''

    f, t, seg_stft = stft(x,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

    #seg_stft = librosa.stft(x, n_fft=nparseg, hop_length=noverlap)

    output = np.abs(seg_stft)

    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output,phase), axis=-3)

    if cut_dc:
        output = output[:,1:,:]

    if cut_last_timeframe:
        output = output[:,:,:-1]

    #return np.rot90(np.abs(seg_stft))
    return output

def matrix_to_submission_task2(classes,locations,length=60.0):
    '''
    Process SELDNet output matrix and create submittable list of frame-wise
    active sounds in the form
    [frame, sound_class, x, y, z]
    '''
    classes=np.reshape(classes,(classes.shape[1],classes.shape[2]))
    locations=np.reshape(locations,(locations.shape[1],locations.shape[2]))
    len_step=length/(classes.shape[0])
    tot_numpy=[]
    for i, (c, l) in enumerate(zip(classes, locations)):
        time=np.asarray([i,len_step*i,len_step*(i+1)])
        cl=np.concatenate((time,c,l),axis=-1)
        tot_numpy.append(cl)
    tot_numpy=np.asarray(tot_numpy)
    pd.DataFrame(tot_numpy).to_csv("foo.csv",header=False,index=False)

    return tot_numpy

def get_label_task2(path,frame_len,file_size,sample_rate,classes_,num_frames,
                    max_label_distance):
    '''
    Process input csv file and output a matrix containing
    the class ids of all sounds present in each data-point and
    their location coordinates, divided in 100 milliseconds frames.
    '''

    class_vec=[]
    loc_vec=[]
    num_classes=len(classes_)
    for k in range(num_frames):
        class_vec.append([None]*num_classes*3)
        loc_vec.append([None]*(num_classes*9))

    class_dict={}
    df=pd.read_csv(path,index_col=False)
    classes_names=classes_
    #classes_names=df['sound_event_recording'].unique()
    #classes_names=classes_names.tolist()
    data_for_class=[]
    for class_name in classes_names:
        class_dict[class_name]=[]
    for class_name in classes_names:
        #data_for_class=df.loc[df['sound_event_recording'] == class_name]
        data_for_class=df.loc[df['Class'] == class_name]
        data_for_class=data_for_class.values.tolist()
        for data in data_for_class:
            class_dict[data[3]].append([data[1],data[2],data[4],data[5],data[6]])

    for j, i in enumerate(np.arange(frame_len,file_size+frame_len,frame_len)):
        start=i-frame_len
        end=i
        for clas in class_dict:
            data_list=class_dict[clas]
            for data in data_list:
                is_in=False
                start_audio=data[0]
                end_audio=data[1]
                x_audio=data[2]
                y_audio=data[3]
                z_audio=data[4]
                if start_audio<start and end_audio>end:
                    is_in=True
                elif end_audio>start and end_audio<end:
                    is_in=True
                elif start_audio>start and start_audio<end:
                    is_in=True
                if is_in:
                    ind_class=classes_names.index(clas)*3
                    if class_vec[j][ind_class]==None:
                        class_vec[j][ind_class]=1
                    elif class_vec[j][ind_class]!=None and class_vec[j][ind_class+1]==None:
                        class_vec[j][ind_class+1]=1
                    else:
                        class_vec[j][ind_class+2]=1
                    if loc_vec[j][ind_class*3]==None:
                        loc_vec[j][0+int(ind_class*3)]=x_audio
                        loc_vec[j][1+int(ind_class*3)]=y_audio
                        loc_vec[j][2+int(ind_class*3)]=z_audio
                    elif loc_vec[j][ind_class*3]!=None and loc_vec[j][ind_class+1]==None:
                        loc_vec[j][3+int(ind_class*3)]=x_audio
                        loc_vec[j][4+int(ind_class*3)]=y_audio
                        loc_vec[j][5+int(ind_class*3)]=z_audio
                    else:
                        loc_vec[j][6+int(ind_class*3)]=x_audio
                        loc_vec[j][7+int(ind_class*3)]=y_audio
                        loc_vec[j][8+int(ind_class*3)]=z_audio
        for i in range(len(loc_vec[j])):
            if loc_vec[j][i]==None:
                loc_vec[j][i]=0
        for i in range(len(class_vec[j])):
            if class_vec[j][i]==None:
                class_vec[j][i]=0

    class_vec = np.array(class_vec)
    loc_vec = np.array(loc_vec)

    loc_vec = loc_vec / max_label_distance  #normalize xyz (to use tanh in the model)

    stacked = np.zeros((class_vec.shape[0],class_vec.shape[1]+loc_vec.shape[1]))
    stacked[:,:class_vec.shape[1]] = class_vec
    stacked[:,class_vec.shape[1]:] = loc_vec
    #return (class_vec), (loc_vec)
    return stacked


def segment_waveforms(predictors, target, length):
    '''
    segment input waveforms into shorter frames of
    predefined length. Output lists of cut frames
    - length is in samples
    '''

    def pad(x, d):
        pad = np.zeros((x.shape[0], d))
        pad[:,:x.shape[-1]] = x
        return pad

    cuts = np.arange(0,predictors.shape[-1], length)  #points to cut
    X = []
    Y = []
    for i in range(len(cuts)):
        start = cuts[i]
        if i != len(cuts)-1:
            end = cuts[i+1]
            cut_x = predictors[:,start:end]
            cut_y = target[:,start:end]
        else:
            end = predictors.shape[-1]
            cut_x = pad(predictors[:,start:end], length)
            cut_y = pad(target[:,start:end], length)
        X.append(cut_x)
        Y.append(cut_y)
    return X, Y


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


def gen_fake_task1_dataset():
    l = []
    target = []
    for i in range(4):
        n = 160000
        n_target = 160000
        sig = np.random.sample(n)
        sig_target = np.random.sample(n_target).reshape((1, n_target))
        target.append(sig_target)
        sig = np.vstack((sig,sig,sig,sig))
        l.append(sig)

    output_path = '../prova_pickle'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path,'training_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'training_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(output_path,'validation_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'validation_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(output_path,'test_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'test_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    '''
    np.save(os.path.join(output_path,'training_predictors.npy'), l)
    np.save(os.path.join(output_path,'training_target.npy'), l)
    np.save(os.path.join(output_path,'validation_predictors.npy'), l)
    np.save(os.path.join(output_path,'validation_target.npy'), l)
    np.save(os.path.join(output_path,'test_predictors.npy'), l)
    np.save(os.path.join(output_path,'test_target.npy'), l)
    '''

    with open(os.path.join(output_path,'training_predictors.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(output_path,'training_target.pkl'), 'rb') as f:
        data2 = pickle.load(f)

    print (data[0].shape)
    print (data2[0].shape)
