# 1. BiLSTM과 CNN

## 1.1. CNN input shape 맞추기 

## 1.2. Subclassing API
### a. Conv1D with Timedistributed layer
```python
class TestModel (tf.keras.Model):
    def __init__(self,vocab_size,tag_size):
        super().__init__()
        self.WordEmbedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)
        self.CharEmbedding = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(len(char_to_index),30,embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')
        self.CharConv1D = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))
        self.CharDropout1 = tf.keras.layers.Dropout(0.5)
        self.CharMaxpool = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(52))
        self.CharFlatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.CharDropout2 = tf.keras.layers.Dropout(0.5)
        self.fBiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), merge_mode='concat')
        self.fTimeDistributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tag_size, activation='softmax'))

    def call(self, inp):
        wrd = self.WordEmbedding(inp[0])
        chrc = self.CharEmbedding(inp[1])
        chrc = self.CharDropout1(chrc)
        chrc = self.CharConv1D(chrc)
        chrc = self.CharMaxpool(chrc)
        chrc = self.CharFlatten(chrc)
        chrc = self.CharDropout2(chrc)
        x = tf.keras.layers.concatenate([wrd,chrc])
        # x= tf.concat([wrd,chrc],axis=-1)
        x = self.fBiLSTM(x)
        x = self.fTimeDistributed(x)
        return x

```
### b. Conv2D with Timedistributed layer
```python
class TestModel2D (tf.keras.Model):
    def __init__(self,vocab_size,tag_size):
        super().__init__()
        self.WordEmbedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)
        self.CharEmbedding = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(len(char_to_index),30,embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')
        self.CharEmbedding2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(len(char_to_index),30,embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')
        self.CharConv2D = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(kernel_size=3, filters=6, padding='same',activation='tanh', strides=1))
        self.CharDropout1 = tf.keras.layers.Dropout(0.5)
        self.CharMaxpool = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D([52,30]))
        self.CharFlatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.CharDropout2 = tf.keras.layers.Dropout(0.5)
        self.fBiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), merge_mode='concat')
        self.fTimeDistributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tag_size, activation='softmax'))

    def call(self, inp):
        wrd = self.WordEmbedding(inp[0])
        chrc = self.CharEmbedding(inp[1])
        chrc = self.CharEmbedding2(chrc)
        chrc = self.CharDropout1(chrc)
        chrc = self.CharConv2D(chrc)
        chrc = self.CharMaxpool(chrc)
        chrc = self.CharFlatten(chrc)
        chrc = self.CharDropout2(chrc)
        # x = tf.keras.layers.concatenate([wrd,chrc])
        x= tf.concat([wrd,chrc],axis=-1)
        x = self.fBiLSTM(x)
        x = self.fTimeDistributed(x)
        return x

```
### c. Conv1D without Timedistributed layer
```python
class TestModel3 (tf.keras.Model):
    def __init__(self,vocab_size,tag_size):
        super().__init__()
        self.WordEmbedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)
        self.CharDropout1 = tf.keras.layers.Dropout(0.5)
        self.CharConv1D =tf.keras.layers.Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1, input_shape=(70,52))
        self.fBiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), merge_mode='concat')
        self.fTimeDistributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tag_size, activation='softmax'))

    def call(self, inp):
        wrd = self.WordEmbedding(inp[0])
        chrc = self.CharDropout1(inp[1])
        chrc = self.CharConv1D(chrc)
        x = tf.keras.layers.concatenate([wrd,chrc])
        x = self.fBiLSTM(x)
        x = self.fTimeDistributed(x)
        return x
```
### d. Conv2D without Timedistributed layer
```python
class TestModel5 (tf.keras.Model):
    def __init__(self,vocab_size,tag_size):
        super().__init__()
        self.WordEmbedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)
        self.CharEmbedding = tf.keras.layers.Embedding(len(char_to_index),30,embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5), mask_zero=True)
        self.CharDropout1 = tf.keras.layers.Dropout(0.5)
        self.CharConv2D =tf.keras.layers.Conv2D(kernel_size=(3,52), filters=30, padding='same',activation='tanh', strides=1, input_shape=(70,52,30))
        # self.CharMaxpooling =  tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(52))
        self.CharMaxpooling =  tf.keras.layers.MaxPooling2D(pool_size=(1,52))
        #self.CharFlatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        # self.CharFlatten = tf.keras.layers.Flatten(data_format='channels_last')
        self.fBiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True), merge_mode='concat')
        self.fTimeDistributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tag_size, activation='softmax'))

    def call(self, inp):
        wrd = self.WordEmbedding(inp[0])
        chrc = self.CharEmbedding(inp[1])
        chrc = self.CharDropout1(chrc)
        chrc = self.CharConv2D(chrc)
        chrc = self.CharMaxpooling(chrc)
        # chrc = self.CharFlatten(chrc)
        chrc = tf.squeeze(chrc)
        x = tf.keras.layers.concatenate([wrd,chrc])
        x = self.fBiLSTM(x)
        x = self.fTimeDistributed(x)
        return x
```



# 2. CNN 
![Alt text](img/charCNN_charBiLSTM.png)

![Alt text](img/CNN_striding.gif)

![Alt text](img/conv1D.jpg)

![Alt text](img/conv2D.png)
![Alt text](img/conv2D_2.png)

![Alt text](img/conv3D.png)
![Alt text](img/conv123summary.png)