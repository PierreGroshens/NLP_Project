import torch
import pandas as pd
from collections import Counter

from transformers import BertTokenizerFast, BertForSequenceClassification

from transformers import AutoTokenizer
import pandas as pd




class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        

        checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        max_length = 280

        #train_encodings = tokenizer(str(train_dataset.text), truncation=True, padding=True, max_length=max_length)



        self.df = pd.read_csv(path)
        self.text = tokenizer(list(self.df.text), padding=True, truncation=True, return_tensors="pt")
        self.label = self.df.label
        
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return (
            self.df.loc[index].text,
            self.df.loc[index].label
            )
            