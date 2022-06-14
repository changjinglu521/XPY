from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from itertools import chain
import tensorflow as tf
import os
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K

path1 = "/home/xiong/PycharmProjects/pythonProject1/data/all_train.csv"
path2 = "/home/xiong/PycharmProjects/pythonProject1/data/dev.csv"



train_label = pd.read_csv(path1,delimiter='\t',header=0,usecols=[1])
train = pd.read_csv(path1,delimiter='\t',header=0,usecols=[0])
train_array = np.array(train)
label_array = np.array(train_label)
train_list = train_array.tolist()
train_label_list = label_array.tolist()
train_x = list(chain.from_iterable(train_label_list))
train_y = list(chain.from_iterable(train_list))
for i in range(len(train_y)):
    if train_y[i] == -1:
        train_y[i]=0

test_label = pd.read_csv(path2,delimiter='\t',header=0,usecols=[1])
test = pd.read_csv(path2,delimiter='\t',header=0,usecols=[0])
test_array = np.array(test)
test_label_array = np.array(test_label)
test_list = test_array.tolist()
test_label_list = test_label_array.tolist()
test_x = list(chain.from_iterable(test_label_list))
test_y = list(chain.from_iterable(test_list))
for i in range(len(test_y)):
    if test_y[i] == -1:
        test_y[i]=0

tokenizer = Tokenizer(num_words=None)
print(type(train_x[0]))
tokenizer.fit_on_texts(train_x)
train_x = tokenizer.texts_to_sequences(train_x)
a = tokenizer.word_index
tokenizer.fit_on_texts(test_x)
test_x = tokenizer.texts_to_sequences(test_x)
b=tokenizer.word_index
word_index = a.copy()
word_index.update(b)

vocab_size=10000        #词库大小
seq_length=300          #句子最大长度
vocab_dim=100          #词的emedding维度
num_classes=2          #分类类别

train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, value=0, padding='post',maxlen=seq_length)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x,value=0, padding='post', maxlen=seq_length)
#print(type(train_x))

###使用预训练的向量
glove_dir = r'/home/xiong/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 2, vocab_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


model = tf.keras.Sequential()#线性叠加层
model.add(layers.Embedding(len(word_index) + 1,
                            vocab_dim,
                            weights=[embedding_matrix],
                            input_length=seq_length))
model.add(layers.Conv1D(filters=256,kernel_size=2,kernel_initializer='he_normal',
                        strides=1,padding='VALID',activation='relu',name='conv'))#一维卷积层
model.add(layers.GlobalMaxPooling1D())#池化层
model.add(layers.Dropout(rate=0.5,name='dropout'))#为了防止过拟合
model.add(layers.Dense(num_classes,activation='softmax'))#全连接网络层
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
history=model.fit(train_x,train_y,epochs=50,batch_size=128,verbose=1,validation_split=0.1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()