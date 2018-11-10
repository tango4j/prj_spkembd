import numpy as np

import torch
import torchtext


parser = argparse.ArgumentParser()

parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')

arg_parsed = parser.parse_args()



