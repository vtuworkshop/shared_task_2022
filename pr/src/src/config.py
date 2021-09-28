import os
import glob
from os.path import *

os.environ['TRANSFORMERS_CACHE'] = 'cache'
from transformers import *

all = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3, 'EXCLAMATION': 4}

# TODO: Change me
SHARE_TASK_BASE='/Users/vietl/projects/vtuworkshop/shared_task_2022'

DATASETS = {
    'pr': {
        'train': glob.glob(join(SHARE_TASK_BASE,'pr/data/train/*.conll')),
        'dev': glob.glob(join(SHARE_TASK_BASE,'pr/data/dev/*.conll')),
        'test': glob.glob(join(SHARE_TASK_BASE,'pr/data/test/*.conll')),
        'n_class': 5,
        'punctuation_dict': all
    },
    'dapr': {
        'train': glob.glob(join(SHARE_TASK_BASE,'dapr/data/train/*.conll')),
        'dev': glob.glob(join(SHARE_TASK_BASE,'dapr/data/dev/*.conll')),
        'test': glob.glob(join(SHARE_TASK_BASE,'dapr/data/test/*.conll')),
        'n_class': 5,
        'punctuation_dict': all
    },
}

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# pretrained src name: (src class, src tokenizer, output dimension, token style)
MODELS = {
    'bert-base-uncased': (BertModel, BertTokenizer, 768, 'bert', 12, BertConfig),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024, 'bert', 24, BertConfig),
    'bert-base-multilingual-cased': (BertModel, BertTokenizer, 768, 'bert', 12, BertConfig),
    'bert-base-multilingual-uncased': (BertModel, BertTokenizer, 768, 'bert', 12, BertConfig),
    'xlm-mlm-en-2048': (XLMModel, XLMTokenizer, 2048, 'xlm', 12, XLMConfig),
    'xlm-mlm-100-1280': (XLMModel, XLMTokenizer, 1280, 'xlm', 16, XLMConfig),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768, 'roberta', 12, RobertaConfig),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024, 'roberta', 24, RobertaConfig),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768, 'bert', 6, DistilBertConfig),
    'distilbert-base-multilingual-cased': (DistilBertModel, DistilBertTokenizer, 768, 'bert', 6, DistilBertConfig),
    'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer, 768, 'roberta', 12, XLMRobertaConfig),
    'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer, 1024, 'roberta', 24, XLMRobertaConfig),
    'albert-base-v1': (AlbertModel, AlbertTokenizer, 768, 'albert', 12, AlbertConfig),
    'albert-base-v2': (AlbertModel, AlbertTokenizer, 768, 'albert', 12, AlbertConfig),
    'albert-large-v2': (AlbertModel, AlbertTokenizer, 1024, 'albert', 24, AlbertConfig),
}
