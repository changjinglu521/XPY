from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from itertools import chain
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


path1 = "/pythonProject1/data/c_1967train.csv"
#path2 = "/home/xiong/PycharmProjects/pythonProject1/data/old/dev.csv"
path3 = "/pythonProject1/data/c_1967test.csv"



train_label = pd.read_csv(path1,delimiter=',',header=0,usecols=[1])
train = pd.read_csv(path1,delimiter=',',header=0,usecols=[0])
train_array = np.array(train)
label_array = np.array(train_label)
train_list = train_array.tolist()
train_label_list = label_array.tolist()
train_x = list(chain.from_iterable(train_label_list))
train_y = list(chain.from_iterable(train_list))
for i in range(len(train_y)):
    if train_y[i] == -1:
        train_y[i]=0

t_label = pd.read_csv(path3,delimiter=',',header=0,usecols=[1])
t_text = pd.read_csv(path3,delimiter=',',header=0,usecols=[0])
t_text_array = np.array(t_text)
t_label_array = np.array(t_label)
t_text_list = t_text_array.tolist()
t_label_list = t_label_array.tolist()
t_x = list(chain.from_iterable(t_label_list))
t_y = list(chain.from_iterable(t_text_list))
for i in range(len(t_y)):
    if t_y[i] == -1:
        t_y[i]=0

all_text = train_x + t_x
all_label = train_y + t_y

print(all_text)

tokenizer = Tokenizer(num_words=None)

tokenizer.fit_on_texts(all_text)
all_text = tokenizer.texts_to_sequences(all_text)


all_label = np.array(all_label)


vocab_size=10000        #词库大小
seq_length=300          #句子最大长度
vocab_dim=100          #词的emedding维度
num_classes=2          #分类类别

all_text = tf.keras.preprocessing.sequence.pad_sequences(all_text, value=0, padding='post',maxlen=seq_length)
#test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x,value=0, padding='post', maxlen=seq_length)
#t_x = tf.keras.preprocessing.sequence.pad_sequences(t_x,value=0, padding='post', maxlen=seq_length)


train_x = all_text[:1730]
train_y = all_label[:1730]
t_x = all_text[1731:-1]
t_y = all_label[1731:-1]


model = tf.keras.Sequential()#线性叠加层
model.add(layers.Embedding(vocab_size, vocab_dim))
model.add(layers.Conv1D(filters=256,kernel_size=2,kernel_initializer='he_normal',
                        strides=1,padding='VALID',activation='relu',name='conv'))#一维卷积层
model.add(layers.GlobalMaxPooling1D())#池化层
model.add(layers.Dropout(rate=0.5,name='dropout'))#为了防止过拟合
model.add(layers.Dense(num_classes,activation='softmax'))#全连接网络层
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['acc'])
history=model.fit(train_x,train_y,epochs=50,batch_size=64,verbose=1,shuffle=True,validation_split=0.1)
plt.plot(history.history['val_acc'])
plt.plot(history.history['val_loss'])
plt.legend(['val_acc', 'loss'], loc='upper left')
plt.show()

from tensorflow.python.keras.utils import plot_model
#from keras.utils import plot_model
plot_model(model,to_file="TextCNN.png",show_shapes=True)


loss, accuracy = model.evaluate(t_x, t_y)
print('Test loss:', loss)
print('Accuracy:', accuracy)