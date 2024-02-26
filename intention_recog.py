
import torch
from torch import nn
from transformers import BertModel , BertTokenizer
from transformers import logging
logging.set_verbosity_error()


class TextClassifier(nn.Module):

    def __init__(self, dropout=0.2):

        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(768, 8)
        self.relu_1 = nn.ReLU()

    def forward(self, seq):
        _, pooled_output = self.bert(input_ids= seq.get('input_ids').squeeze(1), attention_mask=seq.get('attention_mask'),return_dict=False)
        linear_output = self.linear_1(pooled_output)
        final_layer = self.relu_1(linear_output)
        return final_layer


class intention_recognition():

    def __init__(self) -> None:
        self.model = TextClassifier()
        self.model.load_state_dict(torch.load('IntentionRecognition_model_cpu.pt', map_location='cpu'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        print('intention recognition model has been loaded successfuly')
    

    def parse_input(self,input_seq):
        x = self.tokenizer(input_seq,padding='max_length', max_length = 512,
                        truncation=True, return_tensors="pt")
        return {'input_ids':x['input_ids'],'attention_mask':x['attention_mask']}


    def go(self,seq):
        seq = self.parse_input(seq)
        mask = seq['attention_mask']
        input_ids = seq['input_ids'].squeeze(1)
        seq={'input_ids':input_ids,'attention_mask':mask}
        prediction = self.model(seq)
        max_index = torch.argmax(prediction)
        print('######################')
        print(f'class:{max_index+1}')
    
