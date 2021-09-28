import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing

from src.argparser import parse_arguments
from src.dataset import ConllDataset
from src.model import *
from src.config import DATASETS
import src.augmentation as augmentation

# torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del
token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length
aug_type = args.augment_type

print('-' * 40)
for k, v in args.__dict__.items():
    print(f'{k}:\t{v}')
print('-' * 40)
dataset_class_map = {
    'pr': (ConllDataset, ConllDataset, ConllDataset, ConllDataset),
    'dapr': (ConllDataset, ConllDataset, ConllDataset, ConllDataset),
}

# Datasets
dataset = DATASETS[args.language]
punctuation_dict = dataset['punctuation_dict']
DatasetClass = dataset_class_map[args.language]
train_set = DatasetClass[0](dataset['train'], tokenizer=tokenizer, punctuation_dict=punctuation_dict,
                            sequence_len=sequence_len, token_style=token_style, is_train=True, augment_rate=ar,
                            augment_type=aug_type)
print('Load English dataset: train', len(train_set))

val_set = DatasetClass[1](dataset['dev'], tokenizer=tokenizer, punctuation_dict=punctuation_dict,
                          sequence_len=sequence_len, token_style=token_style, is_train=False)
print('Load English dataset: dev', len(val_set))
test_set_ref = DatasetClass[2](dataset['test'][0], tokenizer=tokenizer, punctuation_dict=punctuation_dict,
                               sequence_len=sequence_len, token_style=token_style, is_train=False)
print('Load English dataset: test', len(test_set_ref))
if len(dataset['test']) > 1:
    test_set_asr = DatasetClass[3](dataset['test'][1], tokenizer=tokenizer, punctuation_dict=punctuation_dict,
                                   sequence_len=sequence_len, token_style=token_style, is_train=False)
    print('Load English dataset: asr', len(test_set_asr))

    test_set = [test_set_ref, test_set_asr]
else:
    test_set = [test_set_ref]

n_class = dataset['n_class']

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 2
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
os.makedirs(args.save_path, exist_ok=True)
log_path = os.path.join(args.save_path, args.name + '_logs.txt')

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_CLASSES = {
    'lstm': DeepPunctuation,
    'trans': TransPunctuation,
}

model = MODEL_CLASSES[args.model](args.pretrained_model, n_class,
                                  freeze_bert=args.freeze_bert,
                                  n_layer=args.n_layer)

model.to(device)
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def evaluate(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    model.eval()
    # +1 for overall result
    tp = np.zeros(1 + len(punctuation_dict), dtype=int)
    fp = np.zeros(1 + len(punctuation_dict), dtype=int)
    fn = np.zeros(1 + len(punctuation_dict), dtype=int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, att, y_mask in data_loader:
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = model(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = model(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp / (tp + fp + 1e-18)
    recall = tp / (tp + fn + 1e-18)
    f1 = 2 * precision * recall / (precision + recall + 1e-18)

    return precision * 100, recall * 100, f1 * 100, 100 * correct / total, cm


def train():
    best_val_f = 0
    acc_iter = args.acc_iter
    for epoch in range(args.epoch):
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        model.train()
        for i, batch in enumerate(train_loader):
            x, y, att, y_mask = batch
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                loss = model.log_likelihood(x, att, y) / acc_iter
                # y_predict = src(x, att, y)
                # y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = model(x, att)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y = y.view(-1)
                loss = criterion(y_predict, y) / acc_iter
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
                correct += torch.sum(y_mask * (y_predict == y).long()).item()

            train_loss += loss.item()
            train_iteration += 1

            loss.backward()

            if i % acc_iter == 0 and i > 0:
                optimizer.step()
                optimizer.zero_grad()
                # print('Passing step')
            y_mask = y_mask.view(-1)
            total += torch.sum(y_mask).item()

        train_loss /= train_iteration
        print('=' * 40)
        log = f'| Epoch: {epoch}, loss={train_loss}'
        print(log)

        p, r, f, acc, cm = evaluate(val_loader)
        for i in range(1, len(punctuation_dict)):
            print(f'| Dev: P={p[i]:.2f}\tR={r[i]:.2f}\tF={f[i]:.2f}')
        model_save_path = os.path.join(args.save_path, f'weights-{epoch}.pt')
        torch.save(model.state_dict(), model_save_path)
        print('-')
        p, r, f, acc, cm = evaluate(test_loaders[0])
        for i in range(1, len(punctuation_dict)):
            print(f'| Ref: P={p[i]:.2f}\tR={r[i]:.2f}\tF={f[i]:.2f}')
        print('-')
        if len(test_loaders) > 1:
            p, r, f, acc, cm = evaluate(test_loaders[1])
            for i in range(1, len(punctuation_dict)):
                print(f'| ASR: P={p[i]:.2f}\tR={r[i]:.2f}\tF={f[i]:.2f}')


if __name__ == '__main__':
    train()
