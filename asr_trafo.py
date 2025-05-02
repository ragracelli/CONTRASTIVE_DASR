#!/usr/bin/env python
# coding: utf-8

# # Automatic Speech Recognition with Transformer
# 
# **Author:** [Apoorv Nandan](https://twitter.com/NandanApoorv)<br>
# **Date created:** 2021/01/13<br>
# **Last modified:** 2021/01/13<br>
# **Description:** Training a sequence-to-sequence Transformer for automatic speech recognition.

# ## Introduction
# 
# Automatic speech recognition (ASR) consists of transcribing audio speech segments into text.
# ASR can be treated as a sequence-to-sequence problem, where the
# audio can be represented as a sequence of feature vectors
# and the text as a sequence of characters, words, or subword tokens.
# 
# For this demonstration, we will use the LJSpeech dataset from the
# [LibriVox](https://librivox.org/) project. It consists of short
# audio clips of a single speaker reading passages from 7 non-fiction books.
# Our model will be similar to the original Transformer (both encoder and decoder)
# as proposed in the paper, "Attention is All You Need".
# 
# 
# **References:**
# 
# - [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
# - [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/abs/1904.13377)
# - [Speech Transformers](https://ieeexplore.ieee.org/document/8462506)
# - [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)

# In[76]:


import os
import csv
import sys

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("String must be 'True' ou 'False'")

if __name__ == "__main__":
    # Checks if at least one argument was provided
    if len(sys.argv) < 2:
        print("Usage: python asr_trafo.py <dataset directory> <dataset type: original, aug, aug_prop, aug_plus_prop><last decoder FFN trainable?> <ultimo decoder trainable?> <last encoder FFN trainable?> <epochs> <file pre treino>")
        sys.exit(1)

    # Obtém os argumentos da linha de comando
    dataset_base = sys.argv[1]
    base_type = sys.argv[2]
    fz_last_dec_ffn = str_to_bool(sys.argv[3])
    fz_last_dec = str_to_bool(sys.argv[4])
    fz_last_enc_ffn = str_to_bool(sys.argv[5])

    epochs = int(sys.argv[6])
    
    # Verifique se o argumento 6 foi fornecido
    if len(sys.argv) > 7:
        pretrained_weights = sys.argv[7]
    else:
        pretrained_weights = ""  # Default value if argument 6 is not provided
    

    
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"

from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import datetime
import random
import jiwer
import keras
from keras import layers
from keras.models import load_model
print(tf.__version__)
#print(keras.__version__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# In[77]:



devices = tf.config.list_physical_devices('GPU')
print(len(devices))  


print(tf.test.is_built_with_cuda())

  

# ## Define the Transformer Input Layer
# 
# When processing past target tokens for the decoder, we compute the sum of
# position embeddings and token embeddings.
# 
# When processing audio features, we apply convolutional layers to downsample
# them (via convolution strides) and process local relationships.

# In[78]:


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


# ## Transformer Encoder Layer

# In[79]:


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                #layers.Dense(feed_forward_dim, activation="relu"),
                #layers.Dense(embed_dim),
                layers.SeparableConv1D(feed_forward_dim, 3, activation='relu', padding='same'),
                layers.SeparableConv1D(embed_dim, 3, padding='same'),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ## Transformer Decoder Layer

# In[90]:


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu", trainable=True),
                layers.Dense(embed_dim, trainable=True),
            ]
        )
        self.ffn.trainable = True

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


# ## Complete the Transformer model
# 
# Our model takes audio spectrograms as inputs and predicts a sequence of characters.
# During training, we give the decoder the target character sequence shifted to the left
# as input. During inference, the decoder uses its own past predictions to predict the
# next token.

# In[91]:


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=5,
        num_layers_dec=3,
        num_classes=34,
    ):
        super().__init__()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = tf.keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            layer_dec = TransformerDecoder(num_hid, num_head, num_feed_forward)
            layer_dec.trainable = fz_last_dec
            if i == num_layers_dec - 1:  # If it is the last layer
                layer_dec.ffn.trainable = fz_last_dec_ffn  # Freezes/unfreezes the feed-forward component
                print(f"Component FFN of the decoder layer {i} FROZEN: {not layer_dec.ffn.trainable}")   
            setattr(
                self,
                f"dec_layer_{i}",
                layer_dec,
            )
            if i == num_layers_dec - 1:  # If it is the last layer
                layer_dec.trainable = fz_last_dec  # Freezes/unfreezes the last decoder layer
            print(f"Decoder layer {i} FROZEN: {not layer_dec.trainable}")   
            
        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        #return [self.loss_metric, self.wra_metric]
        return [self.loss_metric]

    
    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            loss = loss_object(one_hot, preds, sample_weight=mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}
        
    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = loss_object(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)       
        return {"loss": self.loss_metric.result()}
        
    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input


# ## Choose Dataset


if dataset_base == 'LJ':

    save_path = "../datasets/LJSpeech-1.1"
    wav_files = glob("{}/**/*.wav".format(save_path), recursive=True)
    #print(wav_files)
    id_to_transcript = {}
    with open(os.path.join(save_path, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            id = line.strip().split("|")[0]
            text = line.strip().split("|")[2]
            id_to_transcript[id] = text


    def get_data(wav_files, id_to_transcript, maxlen=50):
        """returns mapping of audio paths and transcription texts"""
        data = []
        for w in wav_files:
            id = w.split("/")[-1].split(".")[0]
            if len(id_to_transcript[id]) < maxlen:
                data.append({"audio": w, "text": id_to_transcript[id]})
        return data
        
elif dataset_base == 'UA':

    save_path ="/mnt/u/home/gracelli/databases/uaspeech/control/wav_files/original"
    wav_files = glob("{}/**/*.wav".format(save_path), recursive=True)
    #print(wav_files)
    id_to_transcript = {}
    with open(os.path.join(save_path, "/mnt/u/home/gracelli/databases/uaspeech/control/wav_files/original/transcript/metadata.csv"), encoding="utf-8") as f:
        for line in f:
            id = line.strip().split("|")[0]
            text = line.strip().split("|")[1]
            id_to_transcript[id] = text

    def get_data(wav_files, id_to_transcript, maxlen=50):
        """returns mapping of audio paths and transcription texts"""
        data = []
        for w in wav_files:
            id = w.split("/")[-1].split(".")[0]
            if len(id_to_transcript[id]) < maxlen:
                data.append({"audio": w, "text": id_to_transcript[id]})
        return data
        
elif base_type == 'original':
   
    save_path = "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/original/" + dataset_base
    #saveto_aug = "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/" + base_type + "/" + dataset_base

    # Get all .wav files em ambos os diretÃ³rios
    wav_files = glob("{}/**/*.wav".format(save_path), recursive=True)
    #wavs_aug = glob("{}/**/*.wav".format(saveto_aug), recursive=True)

    # Combine the lists of files .wav
    #wav_files = wav_files + wavs_aug

    # DicionÃ¡rio para armazenar os textos
    id_to_transcript = {}

    # Function to read the transcription files
    def read_transcript(file_path):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                id = line.strip().split("|")[0]
                text = line.strip().split("|")[1]
                id_to_transcript[id] = text

    # Reads the transcription files for databases
    read_transcript(os.path.join(save_path, "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/original/" + dataset_base + "/ua_transcript.txt"))
    #read_transcript(os.path.join(saveto_aug, "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/" + base_type + "/" + dataset_base + "/ua_transcript.txt"))

    def get_data(wav_files, id_to_transcript, maxlen=50):
        """returns mapping of audio paths and transcription texts"""
        data = []
        for w in wav_files:
            id = w.split("/")[-1].split(".")[0]
            if len(id_to_transcript[id]) < maxlen:
                data.append({"audio": w, "text": id_to_transcript[id]})
        return data
    
    
else: 

    save_path = "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/original/" + dataset_base
    #saveto_aug = "/mnt/u/home/gracelli/databases/uaspeech/control/wav_files/" + base_type + "/" + dataset_base

    # Get all .wav files in 
    wav_files = glob("{}/**/*.wav".format(save_path), recursive=True)
    wavs_aug = glob("{}/**/*.wav".format(saveto_aug), recursive=True)

    # Combine the lists of files .wav
    wav_files = wav_files + wavs_aug

    # Dictionary to armazenate texts
    id_to_transcript = {}

    # Function to read transcription files
    def read_transcript(file_path):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                id = line.strip().split("|")[0]
                text = line.strip().split("|")[1]
                id_to_transcript[id] = text

    # Reads the transcription files for databases
    read_transcript(os.path.join(save_path, "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/original/" + dataset_base + "/ua_transcript.txt"))
    read_transcript(os.path.join(saveto_aug, "/mnt/u/home/gracelli/databases/uaspeech/dysarthric/wav_files/" + base_type + "/" + dataset_base + "/ua_transcript.txt"))

    def get_data(wav_files, id_to_transcript, maxlen=50):
        """returns mapping of audio paths and transcription texts"""
        data = []
        for w in wav_files:
            id = w.split("/")[-1].split(".")[0]
            if len(id_to_transcript[id]) < maxlen:
                data.append({"audio": w, "text": id_to_transcript[id]})
        return data


# ## Preprocess the dataset

# In[93]:


class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


max_target_len = 200  # all transcripts in out data are < 200 characters
data = get_data(wav_files, id_to_transcript, max_target_len)
vectorizer = VectorizeChar(max_target_len)
print("vocab size", len(vectorizer.get_vocabulary()))
print(vectorizer.get_vocabulary())


def create_text_ds(data):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    eps = 1e-10
    x = (x - means) / (stddevs + eps)
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
    return audio_ds


def create_tf_dataset(data, bs=4):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)
    train_dataset = tf.data.Dataset.zip((audio_ds, text_ds))
    train_dataset = train_dataset.map(lambda x, y: {"source": x, "target": y})
    train_dataset = train_dataset.batch(bs)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset

# Splitting the data
if dataset_base == 'UA':
    train_data = []
    test_data = []
    for item in data:
        #print(item)
        if "CM06" in item["audio"]:
            test_data.append(item)
            #print(item["audio"])
        else:
            train_data.append(item)
            #print('train', item)
            
    tx_ratio = 1
    train_data = train_data[:int(len(train_data)*tx_ratio)]
    test_data = test_data[:int(len(test_data)*tx_ratio)]
    train_dataset = create_tf_dataset(train_data, bs=64)
    val_dataset = create_tf_dataset(test_data, bs=64)
    
elif dataset_base == 'LJ':

    #for LJ
    split = int(len(data) * 0.99)
    train_data = data[:split]
    test_data = data[split:]
    train_dataset = create_tf_dataset(train_data, bs=64)
    val_dataset = create_tf_dataset(test_data, bs=64)

else:
    '''
    train_data = []
    test_data = []
    for item in data:
        #print(item)
        if "B3" in item["audio"]:
            test_data.append(item)
            #print(item["audio"])
        else:
            train_data.append(item)
            #print('train', item)
            
    tx_ratio = 1
    train_data = train_data[:int(len(train_data)*tx_ratio)]
    test_data = test_data[:int(len(test_data)*tx_ratio)]
    train_dataset = create_tf_dataset(train_data, bs=64)
    val_dataset = create_tf_dataset(test_data, bs=64)

    '''
    train_data = []
    b3_test_dataset = []

    for item in data:
        if "B3" in item["audio"]:
            b3_test_dataset.append(item)
        else:
            train_data.append(item)

    # Limitate for tests (tx_ratio < 1.0), "1.0" uses all dataset
    tx_ratio = 1.0
    train_data = train_data[:int(len(train_data) * tx_ratio)]
    b3_test_dataset = b3_test_dataset[:int(len(b3_test_dataset) * tx_ratio)]

    # Shuffle to ensure random distribution
    import random
    random.shuffle(train_data)

    # Split 80/20 into training and validation
    split_index = int(0.8 * len(train_data))
    train_split = train_data[:split_index]
    val_split = train_data[split_index:]

    # Create datasets
    train_dataset = create_tf_dataset(train_split, bs=64)       # Training
    val_dataset = create_tf_dataset(val_split, bs=64)     # Validation
    test_ds = create_tf_dataset(b3_test_dataset, bs=64) # Test with prefix B3

#print(train_split[0:5])

#print(b3_test_dataset[0:5])


# ## Callbacks to display predictions


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):       
        if epoch % 5 != 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        same=0
        results={}
        wras = []
        max_wra_list = []
        wra = 0
        len_preds = len(preds)  # pre-calcule len(preds)
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}
")
            results[i] = {'target': target_text.replace('-',''), 'prediction': prediction}
            if results[i]['target'] == results[i]['prediction']:
                same += 1
                wra = same/len_preds  # Reuse len_preds
            print('WRA = ', round(wra * 100, 2), '%' )
            wras.append(wra)
            
        max_wra = max(wras)
        max_wra_list = append(max_wra)
        print('The maximum value is: ', max_wra)
        #print('dataset_base: ', dataset_base, 'routine: ',  os.path.basename(__file__))
        print('dataset_base: ', dataset_base, 'routine: ', base_type)
        with open(dataset_base + "_" + base_type + '_wras.csv', 'a') as f:
            for i, wra in enumerate(max_wra_list, start=1):
                f.write(f'{i}, {wra}
')


# ## Learning rate schedule

# In[98]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """linear warm up - linear decay"""
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        epoch = tf.cast(epoch, "float32")
        return self.calculate_lr(epoch)


# ## Create & train the end-to-end model

# In[100]:


batch = next(iter(val_dataset))

# The vocabulary to convert predicted indices into characters
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
)  # set the arguments as per vocabulary index for '<' and '>'

model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=max_target_len,
    num_layers_enc=5,
    num_layers_dec=3,
    num_classes=34,
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1,
)

learning_rate = CustomSchedule(
    init_lr=0.00001,
    lr_after_warmup=0.001,
    final_lr=0.00001,
    warmup_epochs=15,
    decay_epochs=85,
    steps_per_epoch=len(train_dataset),
)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.8, beta_2=0.9)

source_dummy = tf.zeros((1, 2754, 129))
target_dummy = tf.zeros((1, 200), dtype=tf.int32)
_ = model((source_dummy, target_dummy))  # initialize all variables

if pretrained_weights != "":
    pre_train_path = 'ckpt/' + pretrained_weights
    model.encoder.load_weights(pre_train_path)  # <- loads only encoder weights
    print(f"Encoder weights loaded from {pre_train_path}")
    
'''
# Settings
fz_encoder_layers = False  # True to make encoders trainable, False to freeze them
#fz_last_enc_ffn = True  # True to make the last FFN trainable, False to freeze it

# Iterate over all encoder layers except the last one
for i, encoder_layer in enumerate(model.encoder.layers[:-1]):  # Excludes the last layer
    encoder_layer.trainable = fz_encoder_layers  # Freezes or unfreezes the entire layer
    encoder_status = "✅ TRAINABLE" if fz_encoder_layers else "🧊 FROZEN"
    print(f"🧱 The complete encoder layer {i} is {encoder_status}.")

    # Check if the layer has the 'ffn' attribute before accessing it
    if hasattr(encoder_layer, 'ffn'):
        for sub_layer in encoder_layer.ffn.layers:
            sub_layer.trainable = False  # Example: keeps FFNs frozen
        print(f"   🔄 The FFN of encoder layer {i} is 🧊 FROZEN.")

# Handling the last encoder layer
last_encoder_layer = model.encoder.layers[-1]

# Freeze or unfreeze FFN of the last layer
if hasattr(last_encoder_layer, 'ffn'):  # Ensure the 'ffn' attribute exists
    for layer in last_encoder_layer.ffn.layers:
        layer.trainable = fz_last_enc_ffn
        ffn_status = "✅ TRAINABLE" if fz_last_enc_ffn else "🧊 FROZEN"
    print(f"🔄 The FFN of the last encoder layer is {ffn_status}.")

# Configure the entire last layer (optional)
last_encoder_layer.trainable = True  # Can adjust the entire layer as well
print(f"🧱 The last encoder layer is fully ✅ TRAINABLE.")
'''


# ✅ Freezes the FFN of the encoder's last layer (agora que os pesos foram carregados)
last_encoder_layer = model.encoder.layers[-1]
for layer in last_encoder_layer.ffn.layers:
    layer.trainable = fz_last_enc_ffn

if not fz_last_enc_ffn:
    print("❄️ Encoder´s last FFN is FROZEN.")
else:
    print("✅ Encoder´s last FFN is TRAINABLE.")



'''
if pretrained_weights != "":
    #pretrained_weights = 'pre_lj/' + pretrained_weights
    pretrained_weights = 'ckpt/' + pretrained_weights
    model.load_weights(pretrained_weights)
    print(f"Model Loaded {pretrained_weights}") 
'''
model.compile(optimizer=optimizer, loss=loss_fn)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=True, write_images=True)

# === Create folders for each speaker results === #
results_dir = os.path.join('results', dataset_base)
os.makedirs(results_dir, exist_ok=True)
print(f"📂 Results directory created: {results_dir}")

# Path where the model will be saved
checkpoint_path = os.path.join(results_dir, f"{dataset_base}_best_model.ckpt")

checkpoint = ModelCheckpoint(
    checkpoint_path,         # file name inside the folder
    monitor='val_loss',       # metric to monitor
    verbose=1, 
    save_best_only=True,      # saves only the best model
    mode='min',               # 'min' para minimizar a 'val_loss'
    save_weights_only=True    # saves only the model weights
)


'''
checkpoint = ModelCheckpoint(f'{dataset_base}_best_model.ckpt',  # file name
                             monitor='val_loss',  # mÃ©trica para monitorar
                             verbose=1, 
                             save_best_only=True,  # saves only the best model
                             mode='min',  # 'min' para minimizar a 'val_loss'
                             save_weights_only=True)  # saves only the model weights

'''

history = model.fit(train_dataset, validation_data=val_dataset, callbacks=[display_cb, checkpoint, tensorboard_callback], epochs=epochs)
model.summary()
print(dataset_base)

# === Load with best model === #
model.load_weights(checkpoint_path)

# Function to compute WER, CER e WAR
def evaluate_model(model, test_ds, idx_to_char):
    ground_truths = []
    predictions = []

    for batch in val_dataset:
        source = batch["source"]
        target = batch["target"]
        preds = model.generate(source, target_start_token_idx=2)  # <token
        preds = preds.numpy()

        for i in range(len(source)):
            target_text = "".join([idx_to_char[_] for _ in target[i, :]]).replace('-', '').replace('<', '').replace('>', '').strip()
            prediction = ""
            for idx in preds[i, :]:
                prediction += idx_to_char[idx]
                if idx == 3:  # > token
                    break
            prediction = prediction.replace('<', '').replace('>', '').replace('-', '').strip()

            ground_truths.append(target_text)
            predictions.append(prediction)

    return ground_truths, predictions

# Evaluate model
ground_truths, predictions = evaluate_model(model, val_dataset, idx_to_char)

# Calculate metrics
wer = jiwer.wer(ground_truths, predictions)
cer = jiwer.cer(ground_truths, predictions)

# WAR (Word Accuracy Rate)
war = 1 - wer

print(f"✅ WER: {wer:.4f}")
print(f"✅ CER: {cer:.4f}")
print(f"✅ WAR: {war:.4f}")

# Save results to files
import pandas as pd

# Create DataFrame
df_metrics = pd.DataFrame({
    'GroundTruth': ground_truths,
    'Prediction': predictions
})

# Save paths
wer_csv_path = os.path.join(results_dir, f"{dataset_base}_wer.csv")
cer_csv_path = os.path.join(results_dir, f"{dataset_base}_cer.csv")
war_csv_path = os.path.join(results_dir, f"{dataset_base}_war.csv")

wer_txt_path = os.path.join(results_dir, f"{dataset_base}_wer.txt")
cer_txt_path = os.path.join(results_dir, f"{dataset_base}_cer.txt")
war_txt_path = os.path.join(results_dir, f"{dataset_base}_war.txt")

# Save metrics
with open(wer_txt_path, "w") as f:
    f.write(f"WER: {wer:.4f}
")
with open(cer_txt_path, "w") as f:
    f.write(f"CER: {cer:.4f}
")
with open(war_txt_path, "w") as f:
    f.write(f"WAR: {war:.4f}
")

# Save detailed predictions
df_metrics.to_csv(wer_csv_path, index=False)

# Create simple DataFrame simples only CER (entire text)
df_cer = pd.DataFrame({
    'GroundTruth': [" ".join(ground_truths)],
    'Prediction': [" ".join(predictions)],
    'CER': [cer]
})
df_cer.to_csv(cer_csv_path, index=False)

# Create simple DataFrame for WAR
df_war = pd.DataFrame({
    'GroundTruth': [" ".join(ground_truths)],
    'Prediction': [" ".join(predictions)],
    'WAR': [war]
})
df_war.to_csv(war_csv_path, index=False)

print(f"📄 Results saved in {results_dir}")


'''
In practice, you should train for around 100 epochs or more.

Some of the predicted text at or around epoch 35 may look as follows:
```
target:     <as they sat in the car, frazier asked oswald where his lunch was>
prediction: <as they sat in the car frazier his lunch ware mis lunch was>

target:     <under the entry for may one, nineteen sixty,>
prediction: <under the introus for may monee, nin the sixty,>
```
'''
# In[39]:


#model.save_weights('pre_ua_pesos_pc_casa.ckpt')
#model.save('pre_ljspeech.h5', save_format="tf")


# In[ ]:

#Group the speakers' WARs and save in txt file
def compute_word_accuracy(model, test_ds, idx_to_char):
    total_words = 0
    correct_words = 0
    
    for batch in val_dataset:
        source = batch["source"]
        target = batch["target"]
        preds = model.generate(source, target_start_token_idx=2)  # Adjust the start token index as per your vocabulary
        preds = preds.numpy()
        for i in range(len(source)):
            target_text = "".join([idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += idx_to_char[idx]
                if idx == 3:  # Adjust the end token index as per your vocabulary
                    break
            # Split text into words
            target_words = target_text.replace('-', '').split()
            prediction_words = prediction.split()
            # Count correct words
            total_words += len(target_words)
            correct_words += sum(1 for tw, pw in zip(target_words, prediction_words) if tw == pw)

    accuracy = correct_words / total_words if total_words > 0 else 0
    return accuracy
    
model.load_weights(f'results/{dataset_base}/{dataset_base}_best_model.ckpt')

word_accuracy = compute_word_accuracy(model, val_dataset, idx_to_char)
print("Word Accuracy:", word_accuracy)

with open("results.txt", "a") as file:
    file.write(f"{dataset_base}_{base_type}_| Accuracy: {word_accuracy}
")