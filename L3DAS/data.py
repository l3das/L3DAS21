import os, sys
from os import listdir,getcwd
from os.path import isfile, join
from . import audio_processing as dsp
import numpy as np
from .download_gdrive import download_file_from_google_drive
import zipfile
from . import metadata_processing as metap
from tqdm import tqdm

class Dataset:
    def download_dataset(self):
        if not os.path.isdir(self.DATASET_PATH+self.TASK_DIR+self.set_type+'/'):
            print("DOWNLOADING DATASET "+self.TASK_DIR[:-1]+"...")
            if self.set_type=='train/':
                self.id_gd='1sXmll-vZstAfdYU2iNd5_nIkCzDRoVk1'
            elif self.set_type=='dev/':
                self.id_gd='1sXmll-vZstAfdYU2iNd5_nIkCzDRoVk1'
            elif self.set_type=='test/':
                self.id_gd='1sXmll-vZstAfdYU2iNd5_nIkCzDRoVk1'
            zip_name='/'+self.TASK_DIR[:-1]+'_'+self.set_type[:-1]+'.zip'
            download_file_from_google_drive(self.id_gd,self.DATASET_PATH+zip_name)
            print("DATASET "+self.TASK_DIR[:-1]+" HAS BEEN DOWNLOADED")
            print("UNZIP THE FOLDER...")
            with zipfile.ZipFile(self.DATASET_PATH+self.TASK_DIR[:-1]+'_'+self.set_type[:-1]+'.zip', 'r') as zip_ref:
                zip_ref.extractall(self.DATASET_PATH)
            print("UNZIPPED")
            os.remove(self.DATASET_PATH+self.TASK_DIR[:-1]+'_'+self.set_type[:-1]+'.zip')
            if not os.path.isdir(self.DATASET_PATH+self.TASK_DIR):
                os.mkdir(self.DATASET_PATH+self.TASK_DIR)
            os.rename(self.DATASET_PATH+'L3DAS_'+self.TASK_DIR[:-1]+'_'+self.set_type[:-1]+'', self.DATASET_PATH+self.TASK_DIR+self.set_type)

    def get_input(self,path):
        audio, sample_rate=dsp.read_audio(path)
        self.sample_rate=sample_rate
        audio=dsp.audio_padding(audio,self.file_size,self.frame_len,int(sample_rate))
        audio,self.frames_per_sample=dsp.split_audio(self.frame_len,int(sample_rate),audio)
        return audio


    def get_output(self,path):
        return metap.get_label(path,self.frame_len,self.file_size,self.sample_rate,self.classes,self.frames_per_sample)

    '''
    INITIALIZATION OF THE CLASS DATASET
    self.name:        name of the chosen dataset, can be Task1 or Task2
    self.frame_len:   length in seconds of a single frame
    self.num_samples: number of audio file to be loaded
    self.input1:      this data will represents the first input data, for example in Task2 it will be the audio file relative to micA
    self.input2:      this data will represents the second input data, for example in Task2 it will be the audio file relative to micB
    self.output1:     this data will represents the first target data, for example in Task2 it will be the vector that indicates if a class of sound is present or not
    self.output2:     this data will represents the second target data, for example in Task2 it will indicates the location of the audio sources for the classes
    self.file_size:   length in seconds of a singol audio file
    self.sample_rate: sample rate for the given dataset
    self.num_classes: num of class for the given dataset
    '''
    def __init__(self,name,num_samples=10,frame_len=1.0,set_type='train',mod='load',saving_dir='insert_the_path_where_you_wanna_save_the_numpyarrays',mic='A',domain='time',spectrum='mp'):
        self.name=name
        self.frame_len=frame_len
        self.num_samples=num_samples
        self.input1=[]
        self.input2=[]
        self.output1=[]
        self.output2=[]
        self.file_size=0
        self.sample_rate=0
        self.classes=0
        self.frames_per_sample=1
        self.set_type=set_type
        self.mod=mod
        self.saving_dir=saving_dir
        self.mic=mic
        self.DATASET_PATH='L3DAS/DATASETS/'
        self.TASK_DIR='Task1/'
        self.INPUT_PATH='data/'
        self.OUTPUT_PATH='labels/'
        self.id_gd='1sXmll-vZstAfdYU2iNd5_nIkCzDRoVk1'
        self.domain=domain
        self.spectrum=spectrum

        if mic not in ['A','AB']:
            print('ERROR: mic can be \'A\' to use the files of mic A and  \'AB\' to use the files of mic A and mic B on the last axis')
            exit()
        if mod not in ['load','save']:
            print('ERROR: mod can be \'load\' to load the data in the RAM memory and  \'save\' to write the preprocessed numpy array in \'saving_dir\'')
            exit()
        if self.domain not in ['time','freq']:
            print('ERROR: domain can be \'time\' to get the signal as numpy array and  \'freq\' to get the spectrum as numpy array')
            exit()
        if self.spectrum not in ['m','p','mp','s']:
            print('ERROR: out can be \'m\' to get the magnitude,  \'p\' to get the phase,  \'mp\' to get magnitude and phase,  \'s\' to get the spectrum concatenating magnitude and phase on the last axis')
            exit()

        if self.name=='Task1':
            self.TASK_DIR='Task1/'
            self.id_gd='1sXmll-vZstAfdYU2iNd5_nIkCzDRoVk1'
            if self.set_type not in ['train','dev','test']:
                print('set_type should be \'train\' or \'dev\' or \'test\'')
            else:
                self.set_type=self.set_type+'/'

            self.download_dataset()
            self.file_size=60.0
            audio_files = [f for f in listdir(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH) if isfile(join(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH, f))]
            audio_filesA = [f for f in audio_files if 'A' in f]
            audio_filesB = [f for f in audio_files if 'B' in f]
            label_files = [f for f in listdir(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH) if isfile(join(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH, f))]
            label_filesA = [f for f in label_files if 'A' in f]
            label_filesB = [f for f in label_files if 'B' in f]
            count=1
            with tqdm(total=self.num_samples) as pbar:
                for audioA, audioB, labelA, labelB  in (zip(sorted(audio_filesA),sorted(audio_filesB), sorted(label_filesA), sorted(label_filesB))):
                    self.input1.append(self.get_input(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH+audioA))
                    self.output1.append(self.get_input(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH+labelA))
                    if mic=='AB':
                        self.input2.append(self.get_input(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH+audioB))
                        self.output2.append(self.get_input(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH+labelB))
                    pbar.update(1)
                    if mod=='save':
                        self.input1=np.asarray(self.input1)
                        self.output1=np.asarray(self.output1)
                        if mic=='AB':
                            self.input2=np.asarray(self.input2)
                            self.output2=np.asarray(self.output2)
                        if domain=='freq':
                            self.input1=dsp.fft_set(self.input1,out=spectrum)
                            self.output1=dsp.fft_set(self.input1,out=spectrum)
                            if mic=='AB':
                                self.input2=dsp.fft_set(self.input2,out=spectrum)
                                self.output2=dsp.fft_set(self.output2,out=spectrum)

                        np.save(self.saving_dir+'/'+audioA[:-3]+'npy',self.input1)
                        np.save(self.saving_dir+'/'+labelA[:-3]+'npy',self.output1)
                        if mic=='AB':
                            np.save(self.saving_dir+'/'+audioB[:-3]+'npy',self.input2)
                            np.save(self.saving_dir+'/'+labelB[:-3]+'npy',self.output2)
                        self.input1=[]
                        self.input2=[]
                        self.output1=[]
                        self.output2=[]
                    if count==self.num_samples:
                        break
                    count+=1
            print('DONE')

        elif self.name=='Task2':
            self.TASK_DIR='Task2/'
            self.id_gd='1sXmll-vZstAfdYU2iNd5_nIkCzDRoVk1'
            if self.set_type not in ['train','dev','test']:
                print('ERROR: set_type can be \'train\' or \'dev\' or \'test\'')
            else:
                self.set_type=self.set_type+'/'

            self.classes=['Chink_and_clink','Computer_keyboard','Cupboard_open_or_close','Drawer_open_or_close','Female_speech_and_woman_speaking','Finger_snapping','Keys_jangling','Knock','Laughter','Male_speech_and_man_speaking','Printer','Scissors','Telephone','Writing']
            #self.download_dataset()
            self.file_size=60.0
            audio_files = [f for f in listdir(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH) if isfile(join(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH, f))]
            audio_filesA = [f for f in audio_files if 'A' in f]
            audio_filesB = [f for f in audio_files if 'B' in f]
            label_files = [f for f in listdir(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH) if isfile(join(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH, f))]
            count=1
            print('LOADING SAMPLES...')
            with tqdm(total=self.num_samples) as pbar:
                for audioA, audioB, label  in (zip(sorted(audio_filesA),sorted(audio_filesB), sorted(label_files))):
                    self.input1.append(self.get_input(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH+audioA))
                    if mic=='AB':
                        self.input2.append(self.get_input(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.INPUT_PATH+audioB))
                    self.output1.append(self.get_output(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH+label)[0])
                    self.output2.append(self.get_output(self.DATASET_PATH+self.TASK_DIR+self.set_type+self.OUTPUT_PATH+label)[1])
                    pbar.update(1)
                    if mod=='save':
                        self.input1=np.asarray(self.input1)
                        self.output1=np.asarray(self.output1)
                        self.output2=np.asarray(self.output2)
                        if mic=='AB':
                            self.input2=np.asarray(self.input2)
                        if domain=='freq':
                            self.input1=dsp.fft_set(self.input1,out=spectrum)
                            if mic=='AB':
                                self.input2=dsp.fft_set(self.input2,out=spectrum)
                        np.save(self.saving_dir+'/'+audioA[:-3]+'npy',self.input1)
                        np.save(self.saving_dir+'/'+label[:-4]+'_class.npy',self.output1)
                        np.save(self.saving_dir+'/'+label[:-4]+'_loc.npy',self.output2)
                        if mic=='AB':
                            np.save(self.saving_dir+'/'+audioB[:-3]+'npy',self.input2)
                        self.input1=[]
                        self.input2=[]
                        self.output1=[]
                        self.output2=[]
                    if count==self.num_samples:
                        break
                    count+=1
            print('DONE')
        else:
            print('ERROR: dataset '+self.name+' doesn\'t exists. Try name= \"Task1\" or \"Task2\"')
            exit()

    def get_dataset(self):
        if self.name=='Task1':
            if self.mic=='A':
                if self.domain=='freq':
                    self.input1=dsp.fft_set(self.input1,out=self.spectrum)
                    self.output1=dsp.fft_set(self.output1,out=self.spectrum)
                return np.asarray(self.input1), np.asarray(self.output1)
            if self.mic=='AB':
                self.input1=np.asarray(self.input1)
                self.input2=np.asarray(self.input2)
                self.output1=np.asarray(self.output1)
                self.output2=np.asarray(self.output2)
                if self.domain=='freq':
                    self.input1=dsp.fft_set(self.input1,out=self.spectrum)
                    self.output1=dsp.fft_set(self.output1,out=self.spectrum)
                    self.input2=dsp.fft_set(self.input2,out=self.spectrum)
                    self.output2=dsp.fft_set(self.output2,out=self.spectrum)
                return np.concatenate((self.input1,self.input2),axis=3), np.concatenate((self.output1,self.output2),axis=3)
        if self.name=='Task2':
            if self.mic=='A':
                if self.domain=='freq':
                    self.input1=dsp.fft_set(self.input1,out=self.spectrum)
                return np.asarray(self.input1), np.asarray(self.output1), np.asarray(self.output2)
            if self.mic=='AB':
                self.input1=np.asarray(self.input1)
                self.input2=np.asarray(self.input2)
                if self.domain=='freq':
                    self.input1=dsp.fft_set(self.input1,out=self.spectrum)
                    self.input2=dsp.fft_set(self.input2,out=self.spectrum)
                return np.concatenate((self.input1,self.input2),axis=3), np.asarray(self.output1), np.asarray(self.output2)
