# -Keras-LSTM-
利用 Keras 下的 LSTM 进行情感分析
我们用 Keras 提供的 LSTM 层构造和训练一个 many-to-one 的 RNN。 网络的输入是一句话，输出是一个情感值（积极或消极）。 所用数据来自 Kaggle 的情感分类比赛 （https://inclass.kaggle.com/c/si650winter11）。 该训练数据长这样： 
1    I either LOVE Brokeback Mountain or think it’s great that homosexuality is becoming more acceptable!: 
1    Anyway, thats why I love ” Brokeback Mountain. 
1    Brokeback mountain was beautiful… 
0    da vinci code was a terrible movie. 
0    Then again, the Da Vinci code is super shitty movie, and it made like 700 million. 
0    The Da Vinci Code comes out tomorrow, which sucks. 
其中的每个句子都有个标签 1 或 0， 用来代表积极或消极。(下载数据) 


 

    先把用到的包一次性全部导入 

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk  #用来分词
import collections  #用来统计词频
import numpy as np
1
2
3
4
5
6
7
8
9
数据准备

     在开始前，先对所用数据做个初步探索。特别地，我们需要知道数据中有多少个不同的单词，每句话由多少个单词组成。 

maxlen = 0  #句子最大长度
word_freqs = collections.Counter()  #词频
num_recs = 0 # 样本数
with open('./train.txt','r+') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))
1
2
3
4
5
6
7
8
9
10
11
12
13
14
     max_len 42 
     nb_words 2324 

      可见一共有 2324 个不同的单词，包括标点符号。每句话最多包含 42 个单词。 
      根据不同单词的个数 (nb_words)，我们可以把词汇表的大小设为一个定值，并且对于不在词汇表里的单词，把它们用伪单词 UNK 代替。 根据句子的最大长度 (max_lens)，我们可以统一句子的长度，把短句用 0 填充。 
      依前所述，我们把 VOCABULARY_SIZE 设为 2002。包含训练数据中按词频从大到小排序后的前 2000 个单词，外加一个伪单词 UNK 和填充单词 0。 最大句子长度 MAX_SENTENCE_LENGTH 设为40。 

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
1
2

      接下来建立两个 lookup tables，分别是 word2index 和 index2word，用于单词和数字转换。 

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
1
2
3
4
5
6

      下面就是根据 lookup table 把句子转换成数字序列了，并把长度统一到 MAX_SENTENCE_LENGTH， 不够的填 0 ， 多出的截掉。 

X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('./train.txt','r+') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17

      最后是划分数据，80% 作为训练数据，20% 作为测试数据。 

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
1
网络构建

      数据准备好后，就可以上模型了。这里损失函数用 binary_crossentropy， 优化方法用 adam。 至于 EMBEDDING_SIZE , HIDDEN_LAYER_SIZE , 以及训练时用到的BATCH_SIZE 和 NUM_EPOCHS 这些超参数，就凭经验多跑几次调优了。 

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
1
2
3
4
5
6
7
8
9
网络训练

      网络构建好后就是上数据训练了。用 10 个 epochs 和 batch_size 取 32 来训练这个网络。在每个 epoch， 我们用测试集当作验证集。 

BATCH_SIZE = 32
NUM_EPOCHS = 10
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))
1
2
3

      Train on 5668 samples, validate on 1418 samples 
      Epoch 1/10 
      5668/5668 [==============================] - 12s - loss: 0.2464 - acc: 0.8897 - val_loss: 0.0672 - val_acc: 0.9697 
      Epoch 2/10 
      5668/5668 [==============================] - 11s - loss: 0.0290 - acc: 0.9896 - val_loss: 0.0407 - val_acc: 0.9838 
      Epoch 3/10 
      5668/5668 [==============================] - 11s - loss: 0.0078 - acc: 0.9975 - val_loss: 0.0506 - val_acc: 0.9866 
      Epoch 4/10 
      5668/5668 [==============================] - 11s - loss: 0.0084 - acc: 0.9970 - val_loss: 0.0772 - val_acc: 0.9732 
      Epoch 5/10 
      5668/5668 [==============================] - 11s - loss: 0.0046 - acc: 0.9989 - val_loss: 0.0415 - val_acc: 0.9880 
      Epoch 6/10 
      5668/5668 [==============================] - 11s - loss: 0.0012 - acc: 0.9998 - val_loss: 0.0401 - val_acc: 0.9901 
      Epoch 7/10 
      5668/5668 [==============================] - 11s - loss: 0.0020 - acc: 0.9996 - val_loss: 0.0406 - val_acc: 0.9894 
      Epoch 8/10 
      5668/5668 [==============================] - 11s - loss: 7.7990e-04 - acc: 0.9998 - val_loss: 0.0444 - val_acc: 0.9887 
      Epoch 9/10 
      5668/5668 [==============================] - 11s - loss: 5.3168e-04 - acc: 0.9998 - val_loss: 0.0550 - val_acc: 0.9908 
      Epoch 10/10 
      5668/5668 [==============================] - 11s - loss: 7.8728e-04 - acc: 0.9996 - val_loss: 0.0523 - val_acc: 0.9901 


      可以看到，经过了 10 个epoch 后，在验证集上的正确率已经达到了 99%。 

网络预测

      我们用已经训练好的 LSTM 去预测已经划分好的测试集的数据，查看其效果。选了 5 个句子的预测结果，并打印出了原句。 

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测','真实','句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
1
2
3
4
5
6
7
8
9
10
      Test score: 0.052, accuracy: 0.990 
      预测 真实 句子 
       0       0      oh , and brokeback mountain is a terrible movie … 
       1       1      the last stand and mission impossible 3 both were awesome movies . 
       1       1      i love harry potter . 
       1       1      mission impossible 2 rocks ! ! … . 
       1       1      harry potter is awesome i do n’t care if anyone says differently ! ..


      可见在测试集上的正确率已达 99%. 

TOY

      我们可以自己输入一些话，让网络预测我们的情感态度。假如我们输入 I love reading. 和 You are so boring. 两句话，看看训练好的网络能否预测出正确的情感。 

INPUT_SENTENCES = ['I love reading.','You are so boring.']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX) ]
label2word = {1:'积极', 0:'消极'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
      积极    I love reading. 
      消极    You are so boring.


  Yes ，预测正确。 

代码

     全部代码如下： 

# -*- coding: utf-8 -*-
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np

## EDA 
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with open('./train.txt','r+') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))

## 准备数据
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('./train.txt','r+') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
## 数据划分
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
## 网络构建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
## 网络训练
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))
## 预测
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测','真实','句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
##### 自己输入
INPUT_SENTENCES = ['I love reading.','You are so boring.']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX) ]
label2word = {1:'积极', 0:'消极'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
