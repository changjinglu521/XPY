import torch
import pandas as pd
from itertools import chain
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
import matplotlib.pyplot as plt
path1 = "/pythonProject1/data/c_Alldata.csv"
#path2 = "/home/xiong/PycharmProjects/pythonProject1/data/dev.csv"

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

#set_seed(1)


model_name = "bert-base-uncased"

max_length = 512

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

train_label = pd.read_csv(path1,delimiter=',',header=0,usecols=[1])
train = pd.read_csv(path1,delimiter=',',header=0,usecols=[0])
train_array = np.array(train)
label_array = np.array(train_label)
train_list = train_array.tolist()
train_label_list = label_array.tolist()
train_t = list(chain.from_iterable(train_label_list))
train_l = list(chain.from_iterable(train_list))
for i in range(len(train_l)):
    if train_l[i] == -1:
        train_l[i]=0

arr = np.arange(1967)
np.random.shuffle(arr)

train_t = np.array(train_t)[arr]
train_l = np.array(train_l)[arr]

train_texts = train_t[:1766]
print(type(train_texts))
train_labels = train_l[:1766]
valid_texts = train_t[1767:1966]
valid_labels = train_l[1767:1966]

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts.tolist(), truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cpu")

from sklearn.metrics import accuracy_score

#计算衡量指标
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=20,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    logging_steps=5,
    evaluation_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,
)
print(trainer.train())
trainer.evaluate()
