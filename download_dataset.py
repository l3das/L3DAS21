import argparse
import os
import wget
import zipfile

'''
Download dataset into a user-defined directory, extracts zip file and delete archive.
Command line arguments through argparse.
'''


def download_l3das_dataset(task, set_type):
    if not os.path.isdir(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+task+set_type+os.sep):
        URL = 'https://zenodo.org/record/4642005/files/'
        zip_name= 'L3DAS_'+task+'_'+set_type+'.zip'
        wget.download(URL+zip_name)
        print("\n")



def extract_dataset(task, set_type):
    if not os.path.isdir(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+task+set_type+os.sep):
        print("UNZIP THE FOLDER...")
        with zipfile.ZipFile(os.getcwd()+os.sep+"L3DAS_"+task+'_'+set_type+'.zip', 'r') as zip_ref:
            zip_ref.extractall(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS")
        print("UNZIPPED")
        os.remove(os.getcwd()+os.sep+"L3DAS_"+task+'_'+set_type+'.zip')
        #if not os.path.isdir(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+task):
        #    os.mkdir(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+task)
        os.rename(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+'L3DAS_'+task+'_'+set_type,
                  os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+task+set_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str,
                    help='which task')
    parser.add_argument('--set_type', type=str,
                        help='which set to download, can be \'train\',\'dev\' or \'test\'')


    '''
    parser.add_argument('--output_path', type=str,
                        help='where to download the dataset')
    '''

    args = parser.parse_args()
    download_l3das_dataset(args.task, args.set_type)
    extract_dataset(args.task, args.set_type)
