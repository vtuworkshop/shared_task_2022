import torch
import torch.nn as nn
from transformers import BertModel


class ChitchatDetection(nn.Module):

    def __init__(self, args):
        super(ChitchatDetection, self).__init__()
        self.device = args.device
        self.bert = BertModel.from_pretrained(args.bert_version)
        hidden_size = self.bert.config.hidden_size

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 2)
        )

    def forward(self, inputs):
        bert_outputs = self.bert(input_ids=inputs['input_ids'].to(self.device),
                                 token_type_ids=inputs['token_type_ids'].to(self.device),
                                 attention_mask=inputs['attention_mask'].to(self.device),
                                 )

        x = bert_outputs['pooler_output']
        logits = self.fc(x)
        return logits
