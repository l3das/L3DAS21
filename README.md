# L3DAS21 challenge supporting API
This repository supports the L3DAS challenge and is aimed at downloading the dataset, pre-processing the sound files and the metadata, training the baseline models and submitting the final results.
We provide easy-to-use instruction to produce the results included in our paper.
Moreover, we extensively commented our base API (contained in the L3DAS folder) for easy customization/re-use of our code.

For further information please refer to the challenge [website](https://sites.google.com/uniroma1.it/l3das/home?authuser=0).



## REQUIREMENTS
Our code is based on Python 3.5.
Required packages:
* torch 1.4.0
* zipp 2.2.0
* librosa 0.8.0
* scipy 1.2.0
* pandas 1.0.3
* numpy 1.18.1
* tqdm 4.45.0
* jiwer 2.2.0
* pystoi 0.3.3
* transformers 4.4.2

To install all dependencies run:
```bash
pip install -r requirements.txt
```
## Dataset Download
To download the L3DAS21 dataset use the script **download_dataset.py**.

Example:
```bash
python3 download_dataset.py --task Task1 --set_type train
```
Options:
* --task: which task's dataset will be downloaded, can be 'Task1' or 'Task2'.
* --set_type: which set to download, can be 'train', 'dev' or 'test'.

## Pre-processing
We provide an automated routine that loads the raw audio waveforms and their correspondent metadata, applies custom pre-processing functions and outputs numpy arrays (.npy files) containing the separate predictors and target matrices.

Run these commands to obtain the matrices needed for our baseline models:
```bash
python3 preprocess_and_save_numpy.py --task Task1 --set_type train --frame_len=20 --domain time --spectrum s --mic AB --num_samples 1 --saving_dir processed

python3 preprocess_and_save_numpy.py --task Task2 --set_type train --frame_len=20 --domain freq --spectrum s --mic AB --num_samples 1 --saving_dir processed
```

Options:
* --task: task to be pre-processed, can be 'Task1' or 'Task2'
* --set_type, set to be pre-processed relative to the given task, can be train, dev or test.
* --frame_len: length in seconds of a single frame
* --domain: domain of the audio sample can be 'time' or 'freq'
* --spectrum: choose what to get from the stft, can be can be 'm' to get the magnitude,  'p' to get the phase,  'mp' to get magnitude and phase,  's' to get the spectrum concatenating magnitude and phase on the last axis. Used only if --domain = 'freq'.
* --saving_dir (str): where to save the processed data
* --mic:  which mic have to be used, mic can be 'A' to use the files of mic A and  'AB' to use the files of mic A and mic B concatenated on the last axis
* --num_samples = num of audio files to process

The two tasks of the challenge require different pre-processing.

For **Task1** the function returns 2 numpy arrays contatining:
* The input multichannel audio waveforms (3d noise+speech scenarios)
* The output monoaural audio waveforms (clean speech)

For **Task2** the function returns 3 numpy arrays contatining:
* The input multichannel audio spectra(3d acoustic scenarios)
* The classes ids of all sounds present in each scene, divided in 100 milliseconds frames
* The location coordinates of all sounds present in each scene, divided in 100 milliseconds frames


## Baseline models training
We will upload this very soon!
