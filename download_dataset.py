
import argparse
import L3DAS

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str,
                    help='which task\'s dataset will be downloaded')
parser.add_argument('--set_type', type=str,
                    help='which set to download, can be \'train\', \'dev\' or \'test\'')

args = parser.parse_args()

L3DAS.data.Dataset(args.task,num_samples=0,set_type=args.set_type)
