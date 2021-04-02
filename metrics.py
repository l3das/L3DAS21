import numpy as np
import csv
import pandas as pd
import torch
import jiwer
import librosa
from pystoi import stoi
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
import sys, os
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h");
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h");

#TASK 1 METRICS
def wer(clean_speech, denoised_speech):
    """
    computes the word error rate(WER) score for 1 single data point
    """
    def _transcription(clean_speech, denoised_speech):

        # transcribe clean audio
        input_values = wer_tokenizer(clean_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_clean = wer_tokenizer.batch_decode(predicted_ids)[0];

        # transcribe
        input_values = wer_tokenizer(denoised_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_estimate = wer_tokenizer.batch_decode(predicted_ids)[0];

        return [transcript_clean, transcript_estimate]

    transcript = _transcription(clean_speech, denoised_speech);
    wer_val = jiwer.wer(transcript[0], transcript[1])

    return wer_val

def task1_metric(clean_speech, denoised_speech, sr=16000):
    '''
    Compute evaluation metric for task 1 as stoi + word error rate
    This function computes such measure for 1 single datapoint
    '''
    WER = wer(clean_speech, denoised_speech)
    STOI = stoi(clean_speech, denoised_speech, sr, extended=False)
    metric = WER + STOI
    return metric, WER, STOI

def task1_average_metric(predicted_folder, truth_folder, fs=16000):
    '''
    Load all submitted sounds for task 1 and compute the average metric
    '''
    metrics = []
    predicted_list = [s for s in os.listdir(predicted_folder) if '.wav' in s]
    truth_list = [s for s in os.listdir(truth_folder) if '.wav' in s]
    n_sounds = len(predicted_list)
    for i in range(n_sounds):
        name = str(i) + '.wav'
        predicted_temp_path = os.path.join(predicted_folder, name)
        truth_temp_path = os.path.join(truth_folder, name)
        predicted = librosa.load(predicted_temp_path, sr=fs)
        truth = librosa.load(truth_temp_path, sr=fs)
        temp_metric, wer, stoi = task1_metric(truth, predicted)
        metrics.append(temp_metric)
        print (predicted_temp_path)

    average_metric = np.mean(metrics)
    print ('Average metric: ', average_metric)

    return average_metric


#TASK 2 METRICS


def location_sensitive_detection(pred_path, true_path,
                                 n_frames=100, spatial_threshold=0.3):
    '''
    Compute TP, FP, FN of a single data point using
    location sensitive detection
    '''
    TP = 0   #true positives
    FP = 0   #false positives
    FN = 0   #false negatives
    #read csv files into numpy matrices
    pred = pd.read_csv(pred_path, sep=',',header=None)
    true = pd.read_csv(true_path, sep=',',header=None)
    pred = pred.values
    true = true.values
    #build empty dict with a key for each time frame
    frames = {}
    for i in range(n_frames):
        frames[i] = {'p':[], 't':[]}
    #fill each time frame key with predicted and true entries for that frame
    for i in pred:
        frames[i[0]]['p'].append(i)
    for i in true:
        frames[i[0]]['t'].append(i)
    #iterate each time frame:
    for frame in range(n_frames):
        t = frames[frame]['t']  #all true events for frame i
        p = frames[frame]['p']  #all predicted events for frame i
        matched = 0           #counts the matching events

        if len(t) == 0:         #if there are PREDICTED but not TRUE events
            FP += len(p)        #all predicted are false positive
        elif len(p) == 0:       #if there are TRUE but not PREDICTED events
            FN += len(t)        #all predicted are false negative

        else:
            for i_t in range(len(t)):           #iterate all true events
                match = False       #flag for matching events
                #count if in each true event there is or not a matching predicted event
                true_class = t[i_t][1]          #true class
                true_coord = t[i_t][-3:]        #true coordinates
                for i_p in range(len(p)):       #compare each true event with all predicted events
                    pred_class = p[i_p][1]      #predicted class
                    pred_coord = p[i_p][-3:]    #predicted coordinates
                    spat_error = np.linalg.norm(true_coord-pred_coord)  #cartesian distance between spatial coords
                    if true_class == pred_class and spat_error < spatial_threshold:  #if predicton is correct (same label + not exceeding spatial error threshold)
                        match = True
                if match:
                    matched += 1    #for each true event, match only once comparing all predicted events

        num_true_items = len(t)
        num_pred_items = len(p)
        fn =  num_true_items - matched
        fp = num_pred_items - matched

        #add to counts
        TP += matched          #number of matches are directly true positives
        FN += fn
        FP += fp

    print ('true positives: ', TP)
    print ('false positives: ', FP)
    print ('false negatives: ', FN)
    print ('---------------------')

    return TP, FP, FN

def compute_seld_metric(predicted_folder, truth_folder, n_frames=100, spatial_threshold=0.3):
    '''
    compute F1 score for the whole set of submitted results based on the
    location sensitive detection metric
    '''
    TP = 0
    FP = 0
    FN = 0
    predicted_list = [s for s in os.listdir(predicted_folder) if '.csv' in s]
    truth_list = [s for s in os.listdir(truth_folder) if '.csv' in s]
    n_files = len(predicted_list)
    #iterrate each submitted file
    for i in range(n_files):
        name = predicted_list[i]
        predicted_temp_path = os.path.join(predicted_folder, name)
        truth_temp_path = os.path.join(truth_folder, name)
        #compute tp,fp,fn for each file
        tp, fp, fn = location_sensitive_detection(predicted_temp_path,
                                                  truth_temp_path,
                                                  n_frames,
                                                  spatial_threshold)
        TP += tp
        FP += fp
        FN += fn

    #compute total F score
    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)

    print ('*******************************')
    F_score = (2 * precision * recall) / (precision + recall + sys.float_info.epsilon)
    print ('F score: ', F_score)
    print ('Precision: ', precision)
    print ('Recall: ', recall)

    return F_score


#gen_dummy_seld_results('./prova')
#compute_seld_metric('./prova/pred', './prova/truth')
