
import argparse
import L3DAS
'''
Take as input the downloaded dataset (audio files and target data)
and output pytorch datasets for task1 and task2, separately.
Command line inputs define which task to process and its parameters
'''

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, help='task to be pre-processed')
parser.add_argument('--set_type', type=str, help='set to be pre-processed relative to the given task')
parser.add_argument('--frame_len', type=float, help='length in seconds of a frame')
parser.add_argument('--domain', type=str, help='domain of the audio file, can be \'time\' or \'freq\'')
parser.add_argument('--spectrum', type=str, default='s', help='choose what to get from the stft, can be can be \'m\' to get the magnitude,  \'p\' to get the phase,  \'mp\' to get magnitude and phase,  \'s\' to get the spectrum concatenating magnitude and phase on the last axis')
parser.add_argument('--saving_dir', type=str, help='where to save the processed data')
parser.add_argument('--mic', type=str, default='A', help='which mic have to be used, mic can be \'A\' to use the files of mic A and  \'AB\' to use the files of mic A and mic B on the last axis')
parser.add_argument('--num_samples', type=int, help='num of audio files to process')
args = parser.parse_args()

L3DAS.data.Dataset(args.task,mod='save',num_samples=args.num_samples,frame_len=args.frame_len,set_type=args.set_type, saving_dir=args.saving_dir, mic=args.mic, domain=args.domain, spectrum=args.spectrum)