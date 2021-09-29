import glob
import json
import argparse
import transformers
import torch
from model import *
from dataset import ChitchatDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='Max epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00002, help='Learning rate')
    parser.add_argument('--bert_version', type=str, default='bert-base-cased', help='Bert version')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--train', type=str, default='../data/train', help='Prefix to the training directory')
    parser.add_argument('--dev', type=str, default='../data/dev', help='Prefix to the dev directory')
    parser.add_argument('--test', type=str, default='../data/test', help='Prefix to the test directory')

    return parser


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_version)
    model = ChitchatDetection(args)
    model.to(device)
    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = ChitchatDataset(args.train, tokenizer)
    dev_dataset = ChitchatDataset(args.dev, tokenizer, sort=True)

    train_dl = DataLoader(train_dataset, shuffle=True, collate_fn=ChitchatDataset.pack,
                          batch_size=args.batch_size,
                          num_workers=4)
    dev_dl = DataLoader(dev_dataset, shuffle=False, collate_fn=ChitchatDataset.pack,
                        batch_size=args.batch_size,
                        num_workers=4)

    for epoch in range(args.epoch):
        model.train()
        for bid, batch in enumerate(train_dl):
            # for k, v in batch.items():
            #     if isinstance(v, torch.Tensor):
            #         print(k, v.shape)
            #     else:
            #         print(k, len(v))
            optimizer.zero_grad()
            logits = model(batch)
            loss = cross_entropy(logits, batch['target'].to(args.device))
            loss.backward()
            optimizer.step()
            if bid % 100 == 0:
                print('Loss: ', loss.detach().cpu().numpy().tolist())

        model.eval()
        with torch.no_grad():
            golden_targets = []
            system_outputs = []
            for bid, batch in enumerate(dev_dl):
                logits = model(batch)
                golden_targets += batch['target'].cpu().numpy().tolist()
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                system_outputs += pred
            ps, rs, fs, ss = precision_recall_fscore_support(golden_targets, system_outputs)
            print(ps, rs, fs)


if __name__ == '__main__':
    args = argument_parser().parse_args()

    for k, v in args.__dict__.items():
        print(k, '\t', v)
    train(args)
