import torch.nn as nn
import torch
from .config import MODELS



class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, n_class, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = n_class
        self.bert = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=n_class)

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        bert_output = self.bert(x, attention_mask=attn_masks, output_hidden_states=True)
        all_states = bert_output['hidden_states']
        x = torch.cat(all_states[-self.n_layers:], dim=-1)
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class TransPunctuation(nn.Module):
    def __init__(self, pretrained_model, n_class, freeze_bert=False, n_layer=1, load_bert=True):
        super(TransPunctuation, self).__init__()
        self.output_dim = n_class
        if load_bert:
            self.bert = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        else:
            from transformers import AutoConfig, AutoModel
            config = AutoConfig.from_pretrained(pretrained_model)
            self.bert = AutoModel.from_config(config)

        self.n_layers = 8
        # Freeze bert layers
        if freeze_bert:
            print('|| BERT IS FROZEN ||')
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            print('|| BERT IS NOT FROZEN ||')
        bert_dim = MODELS[pretrained_model][2]
        self.prj = nn.Linear(bert_dim * self.n_layers, bert_dim)

        l = nn.TransformerEncoderLayer(bert_dim, 8, 1024)
        self.trans = nn.TransformerEncoder(l, n_layer)
        self.linear = nn.Linear(in_features=bert_dim, out_features=n_class)

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        bert_output = self.bert(x, attention_mask=attn_masks, output_hidden_states=True)
        all_states = bert_output['hidden_states']
        x = torch.cat(all_states[-self.n_layers:], dim=-1)
        # (B, N, E) -> (N, B, E)
        x = self.prj(x)
        x = self.trans(x)
        # (N, B, E) -> (B, N, E)
        x = self.linear(x)
        return x
