# L3DAS21 challenge supporting API
This repository supports the L3DAS challenge and is aimed at downloading the dataset, pre-processing the sound files and the metadata, training the baseline models and submitting the final results.
We provide easy-to-use instruction to produce the results included in our paper.
Moreover, we extensively commented our base API for easy customization of our code.

For further information please refer to the challenge [website](https://sites.google.com/uniroma1.it/l3das/home?authuser=0).



## Installation
Our code is based on Python 3.5.

To install all dependencies run:
```bash
pip install -r requirements.txt
```
## Dataset Download
The script **download_dataset.py** is aimed at the dataset download.

Options:
* --task: which task's dataset will be downloaded, can be 'Task1' or 'Task2'.
* --set_type: which set to download. Can be 'train100', 'train360' or dev for task 1 and 'train' or 'dev' for task 2.
* --output_path: where to put and extract the downloaded data

To download all dataset folders run:
```bash
python3 download_dataset.py --task Task1 --set_type train100
python3 download_dataset.py --task Task1 --set_type train360
python3 download_dataset.py --task Task1 --set_type dev
python3 download_dataset.py --task Task2 --set_type train
python3 download_dataset.py --task Task2 --set_type dev
```
These scripts automatically extract the archives and delete the zip files.

## Pre-processing
The file **preprocessing.py** provides automated routines that load the raw audio waveforms and their correspondent metadata, applY custom pre-processing functions and save numpy arrays (.pkl files) containing the separate predictors and target matrices.

Run these commands to obtain the matrices needed for our baseline models:
```bash
python3 preprocessing.py --task Task1 --set_type train --frame_len=20 --domain time --spectrum s --mic AB --num_samples 1 --saving_dir processed

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
