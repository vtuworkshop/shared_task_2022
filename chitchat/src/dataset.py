import torch
import glob
from torch.utils.data import Dataset
import os

CHITCHAT_DICT = {
    'O': 0,
    'CHITCHAT': 1
}


class ChitchatDataset(Dataset):

    def __init__(self, prefix, tokenizer, label_dict=CHITCHAT_DICT, sort=True):
        super(ChitchatDataset, self).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.files = glob.glob1(prefix, '*.conll')
        if sort:
            self.files = sorted(self.files)

        print('Loading: ', len(self.files), ' from ', prefix)

        for file in self.files:
            l = len(self.data)
            full_path = os.path.join(prefix, file)
            with open(full_path) as f:
                lines = f.readlines()
            for line in lines:
                if '\t' in line:
                    parts = line.strip().split('\t')
                    if parts[1] in label_dict:
                        self.data.append({
                            'text': parts[0],
                            'label': parts[1],
                            'target': label_dict[parts[1]]
                        })
            # print(full_path, len(self.data) - l)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        t = self.tokenizer(item['text'],
                           padding='max_length',
                           truncation=True,
                           max_length=64)
        t['text'] = item['text']
        t['label'] = item['label']
        t['target'] = item['target']
        return t

    @staticmethod
    def keep(item):
        return item

    @staticmethod
    def features():
        return {
            'input_ids': torch.LongTensor,
            'token_type_ids': torch.LongTensor,
            'attention_mask': torch.FloatTensor,
            'target': torch.LongTensor,
            'text': ChitchatDataset.keep,
            'label': ChitchatDataset.keep
        }

    @staticmethod
    def pack(items):
        batch = dict()
        # print('Number of items in batch: ', len(items))
        for k, f in ChitchatDataset.features().items():
            batch[k] = f([x[k] for x in items])
        return batch


def test_tokenizer():
    import transformers
    t = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    text = ['this is a sample',
            'this is a much longer example with two clauses']

    print(t(text, padding=True, truncation=True, return_tensors="pt", max_length=5))


if __name__ == '__main__':
    test_tokenizer()
