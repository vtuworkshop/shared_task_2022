import argparse
from .config import DATASETS


def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration')
    parser.add_argument('--name', default='punctuation-restore', type=str, help='name of run')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--pretrained-model', default='roberta-large', type=str, help='pretrained LMs')
    parser.add_argument('--freeze-bert', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Freeze BERT layers or not')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in LM')
    parser.add_argument('--model', default='trans', choices=['lstm', 'lstmcrf', 'lstmwide', 'trans'])
    parser.add_argument('--n-layer', default=1, type=int)
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')

    parser.add_argument('--language', default='pr', type=str,
                        choices=DATASETS.keys(), help="Dataset name")
    parser.add_argument('--acc-iter', default=32, type=int)
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--augment-rate', default=0.15, type=float, help='token augmentation probability')
    parser.add_argument('--augment-type', default='all', type=str, help='which augmentation to use')
    parser.add_argument('--sub-style', default='unk', type=str, help='replacement strategy for substitution augment')
    parser.add_argument('--alpha-sub', default=0.4, type=float, help='augmentation rate for substitution')
    parser.add_argument('--alpha-del', default=0.4, type=float, help='augmentation rate for deletion')
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gradient-clip', default=-1, type=float, help='gradient clipping (default: -1 i.e., none)')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size (default: 8)')
    parser.add_argument('--epoch', default=20, type=int, help='total epochs (default: 10)')
    parser.add_argument('--save-path', default='out-augment/', type=str, help='model and log save directory')

    args = parser.parse_args()
    return args
