import pandas as pd
import re
import torch
import random
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from transformers import BertForSequenceClassification, AdamW
import time
import sys
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)
cuda0 = torch.device('cuda:0')
print("done importing",flush=True)

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer

def get_class_weights(all_labels: list, classes: int):
    """
    Calculate class weight in order to enforce a flat prior
    :param examples:  data examples
    :param label_field_name: a name of label attribute of the field (if e is an Example and a name is "label",
        e.label will be reference to access label value
    :param classes: number of classes
    :return: an array of class weights (cast as torch.FloatTensor)
    """
    #this part is related to this task###### the function can get labels in integer format from beginning
    # print(all_labels)
    all_labels = [0 if item == "disagree" else item for item in all_labels]
    all_labels = [1 if item == "agree" else item for item in all_labels]
    all_labels = [2 if item == "other" else item for item in all_labels]
    # print(all_labels)
    #######################################
    arr = torch.zeros(classes)
    for e in all_labels:
        arr[e] += 1

    arrmax = arr.max().expand(classes)
    return (arrmax / arr).to(cuda0)    

class Bert(Dataset):

  def __init__(self, train_df, val_df):
    self.label_dict = {'disagree': 0,'agree': 1, 'other': 2}

    self.train_df = train_df
    self.val_df = val_df
   

    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained('UBC-NLP/ARBERT')
    self.train_data = None
    self.val_data = None
    self.test_data=None
    self.init_data()

  def init_data(self):
    self.train_data = self.load_data(self.train_df)
    self.val_data = self.load_data(self.val_df)
  

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['s1'].to_list()
    hypothesis_list = df['s2'].to_list()
    label_list = df['stance'].to_list()

    for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False, max_length=512-3-len(premise_id), truncation=True)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      y.append(self.label_dict[label])
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset),flush=True)
    return dataset

  def get_data_loaders(self, batch_size=16, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )
    

    return train_loader, val_loader

#calculate accuracy
def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc


def train(model, train_loader, val_loader, optimizer):
  #early stopping
  last_loss = 100
  patience = 5
  triggertimes = 0  

  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      outputs = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids)
      logits=outputs.logits
      wce_loss = nn.CrossEntropyLoss(weight=class_weights)
      loss = wce_loss(logits, labels) 
      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      

    train_loss = total_train_loss/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        outputs = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids)
        logits=outputs.logits
        wce_loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = wce_loss(logits, labels) 
        total_val_loss += loss.item()

    val_loss = total_val_loss/len(val_loader)
    if val_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times,flush=True)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.',flush=True)
                return model

    else:
            print('trigger times: 0')
            trigger_times = 0

    last_loss = val_loss
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}',flush=True)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds),flush=True)
  return model
#=================================================================================================================================================================
EPOCHS =25
lr_list=[1e-5,2e-5,3e-5]
seeds=[2023]
data="AuSTR"
for seed in seeds:
    
    for fold in [1,2,3,4,5]:
      tuning_df=pd.DataFrame()
      sys.stdout = open("results/%s_seed%s_fold%s_WCE.txt"%(data,seed,fold), "w")
      print("fold",fold)
      tuning_model=[]
      tuning_lr=[]
      tuning_F1=[]

      for lr in lr_list:
       torch.manual_seed(seed) 
       print("lr",lr)
       tuning_lr.append(lr)
       tuning_model.append("ARBERT")
       output_dir="models/%s_seed%s_fold%s_WCE_%lr"%(data,seed,fold,lr)
       train_df = pd.read_csv("data/%s/folds/train_fold%s.txt"%(data,fold),sep="\t")
       val_df = pd.read_csv("data/%s/folds/dev_fold%s.txt"%(data,fold),sep="\t")
       train_df=train_df.dropna().reset_index(drop=True)
       print("size of train",len(train_df))
       val_df=val_df.dropna().reset_index(drop=True)
       print("size of dev",len(val_df))
       print("Done reading data",flush=True)
       class_weights=get_class_weights(train_df['stance'].tolist(),classes=3)
       print("class_weights",class_weights)
       model = BertForSequenceClassification.from_pretrained("UBC-NLP/ARBERT", num_labels=3)
       model.to(device)
       param_optimizer = list(model.named_parameters())
       no_decay = ['bias', 'gamma', 'beta']
       optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
        ]
       optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
       dataset = Bert(train_df, val_df)
       train_loader, val_loader = dataset.get_data_loaders(batch_size=16)
       print("Done processing data and will start training......",flush=True)
       model=train(model, train_loader, val_loader, optimizer)

    
       if not os.path.exists(output_dir):
        os.makedirs(output_dir)
       model.save_pretrained(output_dir)

       #Report results on dev set
       finetuned_model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=3) 
       finetuned_model.to(device)
       probs_all = []
       golden=[]
       with torch.no_grad():
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
         optimizer.zero_grad()
         pair_token_ids = pair_token_ids.to(device)
         mask_ids = mask_ids.to(device)
         seg_ids = seg_ids.to(device)
         labels = y.to(device)

         _, prediction = finetuned_model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        
         probs_all += prediction.tolist()
         golden+=labels.tolist()
       F1_score=f1_score(golden,torch.log_softmax(torch.tensor(probs_all), dim=1).argmax(dim=1).tolist(),average='macro')
       print("F1_score",F1_score,flush=True)
       tuning_F1.append(F1_score)
       print("*****************************************************************************************",flush=True)
      tuning_df['BERT model']=tuning_model
      tuning_df['learning rate']=tuning_lr
      tuning_df['Macro-F1']=tuning_F1
      tuning_df.to_csv("results/tuning_results/%s_seed%s_fold%s_WCE_tuning.txt"%(data,seed,fold),sep="\t")
     #print(accuracy_score(golden,torch.log_softmax(torch.tensor(probs_all), dim=1).argmax(dim=1).tolist()),flush=True)
     #print(f1_score(golden,torch.log_softmax(torch.tensor(probs_all), dim=1).argmax(dim=1).tolist(),average=None),flush=True)
