import argparse
import os
import wget
import zipfile

'''
Download dataset into a user-defined directory
Command line arguments through argparse
'''


def download_l3das_dataset(task, set_type):
    if not os.path.exists(os.getcwd()+os.sep+'L3DAS_'+task+'_'+set_type+'.zip'):
        URL = 'https://zenodo.org/record/4642005/files/'
        zip_name= 'L3DAS_'+task+'_'+set_type+'.zip'
        wget.download(URL+zip_name)
        print("\n")
    else:
        print("EXISTING FOLDER\n")




def extract_dataset(task, set_type,output_path):
    if not os.path.isdir(os.getcwd()+os.sep+output_path+os.sep+task+set_type+os.sep):
        print("UNZIP THE FOLDER...")
        with zipfile.ZipFile(os.getcwd()+os.sep+"L3DAS_"+task+'_'+set_type+'.zip', 'r') as zip_ref:
            zip_ref.extractall(os.getcwd()+os.sep+output_path)
        print("UNZIPPED")
        os.remove(os.getcwd()+os.sep+"L3DAS_"+task+'_'+set_type+'.zip')
        #if not os.path.isdir(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+task):
        #    os.mkdir(os.getcwd()+os.sep+"L3DAS"+os.sep+"DATASETS"+os.sep+task)
        os.rename(os.getcwd()+os.sep+output_path+os.sep+'L3DAS_'+task+'_'+set_type,
                  os.getcwd()+os.sep+output_path+os.sep+task+set_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str,
                    help='which task')
    parser.add_argument('--set_type', type=str,
                        help='which set to download, can be "train100", "train360" or "dev" for task1 and "train" or "dev" for task 2')
    parser.add_argument('--output_path', type=str,
                        default="L3DAS"+os.sep+"DATASETS",
                        help='where to download the dataset',)


    args = parser.parse_args()
    download_l3das_dataset(args.task, args.set_type)
    extract_dataset(args.task, args.set_type, args.output_path)
