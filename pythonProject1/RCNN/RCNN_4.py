from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from itertools import chain
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

path1 = "/pythonProject1/data/c_1967train.csv"
#path2 = "/home/xiong/PycharmProjects/pythonProject1/data/old/dev.csv"
path2 = "/pythonProject1/data/c_1967test.csv"



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
tokenizer.fit_on_texts(test_x)
test_x = tokenizer.texts_to_sequences(test_x)
#print(len(train_y))
#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#把序列补成一样的长度
#data = pad_sequences(sequences, maxlen=40)
#print("========================")

#print(train_y)

vocab_size=10000        #词库大小
seq_length=300          #句子最大长度
vocab_dim=100          #词的emedding维度
num_classes=2          #分类类别

train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, value=0, padding='post',maxlen=seq_length)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x,value=0, padding='post', maxlen=seq_length)
#print(type(train_x))

model = tf.keras.Sequential()#线性叠加层
model.add(layers.Embedding(vocab_size, vocab_dim))
#model.add(layers.Dropout(0.5))#为了防止过拟合
model.add(layers.Bidirectional(layers.LSTM(32,return_sequences=True),merge_mode='concat'))#一维卷积层
model.add(layers.GlobalMaxPooling1D())#池化层
model.add(layers.Dense(num_classes,activation='softmax'))#全连接网络层
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
history=model.fit(train_x,train_y,epochs=50,batch_size=128,verbose=1,validation_split=0.1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()

from tensorflow.python.keras.utils import plot_model
#from keras.utils import plot_model
plot_model(model,to_file="LSTM.png",show_shapes=True)

loss, accuracy = model.evaluate(test_x, test_y)
print('Test loss:', loss)
print('Accuracy:', accuracy)