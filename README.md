# L3DAS21 challenge supporting API
This repository supports the L3DAS21 challenge and is aimed at downloading the dataset, pre-processing the sound files and the metadata, training and evaluating the baseline models and validating the final results.
We provide easy-to-use instruction to produce the results included in our paper.
Moreover, we extensively commented our code for easy customization.

For further information please refer to the challenge [website](https://sites.google.com/uniroma1.it/l3das/home?authuser=0).



## Installation
Our code is based on Python 3.5.

To install all Python dependencies run:
```bash
pip install -r requirements.txt
```
## Dataset download
The script **download_dataset.py** is aimed at the dataset download.

To download all dataset folders run:
```bash
python3 download_dataset.py --task Task1 --set_type train100 --output_path DATASETS/Task1
python3 download_dataset.py --task Task1 --set_type train360 --output_path DATASETS/Task1
python3 download_dataset.py --task Task1 --set_type dev --output_path DATASETS/Task1
python3 download_dataset.py --task Task2 --set_type train --output_path DATASETS/Task2
python3 download_dataset.py --task Task2 --set_type dev --output_path DATASETS/Task2
```
These scripts automatically extract the archives and delete the zip files.

Alternatively, it is possible to manually download the dataset from [zenodo](https://zenodo.org/record/4642005#.YGcX-hMzaAx).

To obtain our baseline results for Task 1, do not download the train360 set. 


## Pre-processing
The file **preprocessing.py** provides automated routines that load the raw audio waveforms and their correspondent metadata, apply custom pre-processing functions and save numpy arrays (.pkl files) containing the separate predictors and target matrices.

Run these commands to obtain the matrices needed for our baseline models:
```bash
python3 preprocessing.py --task 1 --input_path DATASETS/Task1 --training_set train100 --processsing_type waveform --num_mics 1 --segmentation_len 2
python3 preprocessing.py --task 2 --input_path DATASETS/Task2 --processsing_type stft --num_mics 1 --frame_len 100
```
The two tasks of the challenge require different pre-processing.

For **Task1** the function returns 2 numpy arrays contatining:
* The input multichannel audio waveforms (3d noise+speech scenarios)
* The output monoaural audio waveforms (clean speech)

For **Task2** the function returns 2 numpy arrays contatining:
* The input multichannel audio spectra (3d acoustic scenarios)
* The class ids of all sounds present in each data-point and their location coordinates, divided in 100 milliseconds frames.



## Baseline models
We provide baseline models for both tasks, implemented in PyTorch. For task 1 we use a Filter and Sum Network (FaSNet) and for task 2 a SELDNet architecture. Please refer to the challenge paper [link] for additional information about our models.

To train our baseline models with the default arguments run:
```bash
python3 train_baseline_task1.py
python3 train_baseline_task2.py
```

GPU is strongly recommended to avoid very long training times.

To compute the challenge metrics for each task using the trained models run:
```bash
python3 evaluate_baseline_task1.py
python3 evaluate_baseline_task2.py
```
## Evaluaton metrics
Our evaluation metrics for both tasks are included in the **metrics.py** script.
TO BE CONTINUED..

## Submission shape validation
The file **validate_submission.py** provides functions to assess wether the files to be submitted for the 2 tasks have the correct naming, format, and shape.
TO BE CONTINUED... 

