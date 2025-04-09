# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:00:26 2025

@author: ragra
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from sklearn.manifold import TSNE
from glob import glob
import librosa
import random
import gc
import time
import csv
#import tensorflow_addons as tfa


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ==== SEED FIX ==== #
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# ==== AUGMENTAÇÃO E PRÉ-PROCESSAMENTO ==== #
def segment_audio_raw_tensor(audio, silence_threshold=0.001, silence_ratio_range=(0.02, 0.1), max_silence_points=2):
    def _segment_numpy(audio_np):
        abs_audio = np.abs(audio_np)
        active_indices = np.where(abs_audio > silence_threshold)[0]
        if len(active_indices) == 0:
            return audio_np.astype(np.float32)

        region_start = active_indices[0]
        region_end = active_indices[-1]
        region_length = region_end - region_start

        for _ in range(random.randint(1, max_silence_points)):
            min_silence = int(silence_ratio_range[0] * region_length)
            max_silence = int(silence_ratio_range[1] * region_length)
            silence_length = random.randint(min_silence, max_silence)
            if region_length - silence_length > 0:
                silence_start = random.randint(region_start, region_end - silence_length)
            else:
                silence_start = region_start
            silence_end = silence_start + silence_length
            audio_np[silence_start:silence_end] = 0

        return audio_np.astype(np.float32)

    result = tf.numpy_function(_segment_numpy, [audio], tf.float32)
    result.set_shape([None])
    return result


def add_white_noise(audio, noise_level=0.1):
    noise = tf.random.normal(tf.shape(audio), stddev=noise_level)
    return tf.clip_by_value(audio + noise, -1.0, 1.0)

def apply_time_stretch(audio, rate=0.9):
    def _stretch_numpy(x):
        stretched = librosa.effects.time_stretch(x, rate=rate)
        if len(stretched) > len(x):
            stretched = stretched[:len(x)]
        else:
            stretched = np.pad(stretched, (0, len(x) - len(stretched)))
        return stretched.astype(np.float32)

    return tf.numpy_function(_stretch_numpy, [audio], tf.float32)

# ==== UTILITÁRIOS DE ÁUDIO ==== #
def load_audio_raw(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    return tf.squeeze(audio, axis=-1)

'''
@tf.function
def preprocess_audio(audio):
    def _stft_norm(x):
        stft = tf.signal.stft(x, frame_length=200, frame_step=80, fft_length=256)
        mag = tf.math.pow(tf.abs(stft), 0.5)
        mean = tf.reduce_mean(mag, axis=1, keepdims=True)
        std = tf.math.reduce_std(mag, axis=1, keepdims=True)
        x_norm = (mag - mean) / (std + 1e-10)
        x_pad = tf.pad(x_norm, [[0, 2754], [0, 0]])
        return x_pad[:2754, :]

    aug1 = apply_time_stretch(audio, rate=0.9)
    aug2 = add_white_noise(segment_audio_raw_tensor(audio), noise_level=3e-4)

    return _stft_norm(aug1), _stft_norm(aug2)
'''
''' #versao mais light - testar
def preprocess_audio(audio):
    def _stft_norm(x):
        stft = tf.signal.stft(x, frame_length=200, frame_step=80, fft_length=256)
        mag = tf.abs(stft)
        mag = tf.math.pow(mag, 0.5)

        mean = tf.reduce_mean(mag, axis=1, keepdims=True)
        std = tf.math.reduce_std(mag, axis=1, keepdims=True)
        x_norm = (mag - mean) / (std + 1e-10)

        # 🟩 Corte de frequências acima de 80 bins
        x_norm = x_norm[:, :80]  

        # 🟩 Downsampling temporal (440 → 220)
        x_norm = x_norm[::2, :]  

        return x_norm

    orig = _stft_norm(audio)
    aug1 = _stft_norm(apply_time_stretch(audio, rate=0.9))
    aug2 = _stft_norm(add_white_noise(segment_audio_raw_tensor(audio), noise_level=3e-4))
    return orig, aug1, aug2
'''

#@tf.function
def preprocess_audio(audio):
    def _stft_norm(x):
        stft = tf.signal.stft(x, frame_length=200, frame_step=80, fft_length=256)
        mag = tf.math.pow(tf.abs(stft), 0.5)
        mean = tf.reduce_mean(mag, axis=1, keepdims=True)
        std = tf.math.reduce_std(mag, axis=1, keepdims=True)
        x_norm = (mag - mean) / (std + 1e-10)
        #x_pad = tf.pad(x_norm, [[0, 2754], [0, 0]])
        x_pad = tf.pad(x_norm, [[0, 440], [0, 0]])
        #return x_pad[:2754, :]
        return x_pad[:440, :]
        
    orig = _stft_norm(audio)
    aug1 = _stft_norm(apply_time_stretch(audio, rate=0.9))
    aug2 = _stft_norm(add_white_noise(segment_audio_raw_tensor(audio), noise_level=3e-4))
    return orig, aug1, aug2

'''
def load_and_augment(path):
    audio = tf.numpy_function(load_audio_raw, [path], tf.float32)
    audio.set_shape([None])
    x1, x2 = preprocess_audio(audio)
    x1 = tf.expand_dims(x1, axis=0)
    x2 = tf.expand_dims(x2, axis=0)
    return {"aug1": x1, "aug2": x2}
'''
'''
def load_and_augment(path):
    audio = tf.numpy_function(load_audio_raw, [path], tf.float32)
    audio.set_shape([None])
    orig, x1, x2 = preprocess_audio(audio)
    orig = tf.expand_dims(orig, axis=0)
    x1 = tf.expand_dims(x1, axis=0)
    x2 = tf.expand_dims(x2, axis=0)
    return {"orig": orig, "aug1": x1, "aug2": x2}
'''
def load_and_augment(path):
    path = tf.ensure_shape(path, [])  # garante escalar
    path = tf.cast(path, tf.string)   # ✅ converte para string explicitamente

    audio = tf.numpy_function(load_audio_raw, [path], tf.float32)
    audio.set_shape([None])
    orig, x1, x2 = preprocess_audio(audio)
    orig = tf.expand_dims(orig, axis=0)
    x1 = tf.expand_dims(x1, axis=0)
    x2 = tf.expand_dims(x2, axis=0)
    return {"orig": orig, "aug1": x1, "aug2": x2}


def create_augmented_datasets(tipo_base="original", bs=16, val_split=0.2, seed=42):
    #saveto = f"/home/gracelli/databases/uaspeech/control/wavs/{tipo_base}"
    saveto = f"U:/home/gracelli/databases/uaspeech/control/wavs/{tipo_base}"

    wavs = glob(f"{saveto}/**/*.wav", recursive=True)
    wavs.sort()
    random.seed(seed)
    random.shuffle(wavs)

    split_idx = int(len(wavs) * (1 - val_split))
    train_paths = wavs[:split_idx]
    val_paths = wavs[split_idx:]

    def create_dataset(file_list):
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        #ds = ds.map(lambda path: load_and_augment(path), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda path: load_and_augment(tf.cast(path, tf.string)), num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    return create_dataset(train_paths), create_dataset(val_paths)


# ==== ENCODER TRANSFORMER PARA SIMCLR ==== #
class SpeechFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class EncoderTransformer(tf.keras.Model):
    def __init__(self, num_hid=200, num_head=2, num_feed_forward=400, num_layers_enc=5):
        super().__init__()
        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid)
        self.encoder_layers = [
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_enc)
        ]

    def call(self, x):
        x = self.enc_input(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

# ==== MODELO SIMCLR ==== #
class SimCLRModel(tf.keras.Model):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(projection_dim)
        ])

    def call(self, x):
        h = self.encoder(x)
        h = tf.reduce_mean(h, axis=1)
        z = self.projection(h)
        return tf.math.l2_normalize(z, axis=1)

class SimCLRLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size, temperature):
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._create_correlated_mask(batch_size)

    def _create_correlated_mask(self, batch_size):
        N = 2 * batch_size
        mask = tf.ones((N, N), dtype=tf.bool)
        mask = tf.linalg.set_diag(mask, tf.zeros(N, dtype=tf.bool))

        idx = tf.range(batch_size)
        i = tf.stack([idx, idx + batch_size], axis=1)
        j = tf.stack([idx + batch_size, idx], axis=1)

        updates = tf.zeros(batch_size, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(mask, i, updates)
        mask = tf.tensor_scatter_nd_update(mask, j, updates)
        return mask

    def call(self, z_i, z_j):
        batch_size = self.batch_size
        N = 2 * batch_size

        # Concat embeddings
        z = tf.concat([z_i, z_j], axis=0)  # shape: (2N, dim)

        # Similaridade cosseno entre todos os pares
        z_i_exp = tf.expand_dims(z, 1)     # (2N, 1, D)
        z_j_exp = tf.expand_dims(z, 0)     # (1, 2N, D)
        sim = tf.reduce_sum(tf.math.l2_normalize(z_i_exp, axis=-1) * tf.math.l2_normalize(z_j_exp, axis=-1), axis=-1)
        sim /= self.temperature  # shape: (2N, 2N)

        # Diagonais deslocadas -> positivos
        sim_i_j = tf.linalg.diag_part(sim, k=batch_size)
        sim_j_i = tf.linalg.diag_part(sim, k=-batch_size)
        positives = tf.concat([sim_i_j, sim_j_i], axis=0)  # shape: (2N,)

        positives = tf.reshape(positives, (N, 1))
        negatives = tf.boolean_mask(sim, self.mask)  # remove positivos
        negatives = tf.reshape(negatives, (N, -1))

        logits = tf.concat([positives, negatives], axis=1)  # (2N, 1+neg)

        # Labels: 0 para a positiva
        labels = tf.zeros(N, dtype=tf.int32)

        # Usar cross entropy com logits
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss


# ==== TSNE PLOT ==== #
def plot_tsne(embeddings, epoch, save_path_prefix="tsne_epoch"):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    plt.title(f"t-SNE of SimCLR Embeddings - Epoch {epoch}")
    save_path = f"{save_path_prefix}_{epoch}.png"
    plt.savefig(save_path)
    plt.close()

class LAMB(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.01, name="ManualLAMB", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def build(self, var_list):
        self.m = [self.add_slot(v, "m") for v in var_list]
        self.v = [self.add_slot(v, "v") for v in var_list]

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, vars = zip(*grads_and_vars)
        if not hasattr(self, 'm'):
            self.build(vars)

        for i, (grad, var) in enumerate(grads_and_vars):
            if grad is None:
                continue

            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")

            m_t = self.beta1 * m + (1 - self.beta1) * grad
            v_t = self.beta2 * v + (1 - self.beta2) * tf.square(grad)

            m_hat = m_t / (1 - self.beta1)
            v_hat = v_t / (1 - self.beta2)

            update = m_hat / (tf.sqrt(v_hat) + self.epsilon) + self.weight_decay * var

            r1 = tf.norm(var)
            r2 = tf.norm(update)
            trust_ratio = tf.where(
                tf.logical_and(r1 > 0, r2 > 0),
                r1 / r2,
                1.0
            )

            scaled_update = trust_ratio * update
            var.assign_sub(self._get_hyper("learning_rate") * scaled_update)

            m.assign(m_t)
            v.assign(v_t)

        return tf.no_op()

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        })
        return config


class LARS(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0005, eeta=0.001, name="LARS", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eeta = eeta

    def build(self, var_list):
        self.velocities = [self.add_slot(v, "velocity") for v in var_list]

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, vars = zip(*grads_and_vars)
        if not hasattr(self, 'velocities'):
            self.build(vars)

        new_weights = []
        for i, (grad, var) in enumerate(grads_and_vars):
            if grad is None:
                continue

            weight_norm = tf.norm(var)
            grad_norm = tf.norm(grad)
            trust_ratio = tf.where(
                tf.logical_and(weight_norm > 0, grad_norm > 0),
                self.eeta * weight_norm / (grad_norm + 1e-9),
                1.0
            )

            scaled_grad = trust_ratio * grad + self.weight_decay * var

            v = self.get_slot(var, "velocity")
            new_v = self.momentum * v - self._get_hyper("learning_rate") * scaled_grad
            var.assign_add(new_v)
            v.assign(new_v)

        return tf.no_op()

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "eeta": self.eeta
        })
        return config


def get_optimizer(name, lr=1e-3):
    if name.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif name.lower() == "lamb":
        return LAMB(learning_rate=lr, weight_decay_rate=0.01)
    elif name.lower() == "lars":
        return LARS(learning_rate=lr, momentum=0.9, weight_decay=0.0005, eeta=0.001)
    else:
        raise ValueError("Escolha um otimizador válido: 'sgd', 'lamb' ou 'lars'")
'''
class CosineAnnealingScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_epochs, eta_min=1e-5):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.eta_min = eta_min

    def __call__(self, step):
        epoch = tf.cast(step, tf.float32)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * epoch / self.total_epochs))
        lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_decay
        return lr
'''

class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        epoch = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (epoch + 1.0) / self.warmup_epochs
        return tf.cond(epoch < self.warmup_epochs,
                       lambda: warmup_lr,
                       lambda: self.initial_lr)
    
class CosineAnnealingWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, T_0, eta_min=0.0, T_mult=1, last_epoch=-1, verbose=False):
        super().__init__()
        self.initial_lr = initial_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.T_cur = last_epoch
        self.T_i = T_0

    def __call__(self, step):
        if self.last_epoch == -1:
            self.T_cur = step
        else:
            # Cálculo do reinício
            i = 0
            T_i = self.T_0
            while step >= T_i:
                step -= T_i
                i += 1
                T_i *= self.T_mult
            self.T_cur = step
            self.T_i = T_i

        #cosine_decay = 0.5 * (1 + tf.cos(np.pi * self.T_cur / self.T_i))
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * tf.cast(self.T_cur, tf.float32) / tf.cast(self.T_i, tf.float32)))

        lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_decay

        #if self.verbose and self.T_cur == 0:
        #    tf.print(f"[WarmRestart] Learning Rate Restarted: {lr}")

        return lr

# ==== EXECUÇÃO DO TREINAMENTO ==== #
def run_pretrain_simclr(optimizer_name='lars', val_every=5):
    projection_dim = 128
    batch_size = 32
    epochs = 10
    base_lr = 1e-3
    warmup_epochs = 5
    tipo_base = "original"

    print("📁 Carregando base controle UA-Speech...")
    raw_ds, val_ds = create_augmented_datasets(tipo_base=tipo_base, bs=batch_size, val_split=0.2)

    print("🧠 Inicializando encoder Transformer...")
    
    encoder_model = EncoderTransformer(num_hid=200, num_head=2, num_feed_forward=400, num_layers_enc=4)
    try:
        encoder_model.load_weights("pre_lj/melhor_modelo.ckpt").expect_partial()
        print("🏋️ Pesos pré-treinados carregados com sucesso.")
        
        # ✅ Passa um dummy input para buildar o modelo antes de salvar
        dummy_input = tf.random.normal([1, 440, 129])  # Compatível com seus espectrogramas
        _ = encoder_model(dummy_input)
        # ✅ Salvar modelo no formato SavedModel (para Keras 3)
        encoder_model.save("models/encoder_keras3", save_format="tf")
        print("💾 Modelo salvo em formato compatível com Keras 3 em 'models/encoder_keras3/'")

    except Exception as e:
        print(f"⚠️ Erro ao carregar pesos: {e}")

    simclr_model = SimCLRModel(encoder=encoder_model, projection_dim=projection_dim)

    # Warmup scheduler
    def warmup_schedule(epoch):
        return base_lr * (epoch + 1) / warmup_epochs

    # Cosine annealing
    cosine_scheduler = CosineAnnealingWarmRestarts(initial_lr=base_lr, T_0=500, eta_min=0.05)

    # Scheduler combinado
    def combined_schedule(epoch):
        return warmup_schedule(epoch) if epoch < warmup_epochs else cosine_scheduler(epoch - warmup_epochs)

    # Otimizador com callback de atualização manual do LR
    optimizer = get_optimizer(optimizer_name, lr=base_lr)
    criterion = SimCLRLoss(batch_size=batch_size, temperature=0.5)

    print(f"🚀 Iniciando pré-treinamento SimCLR com otimizador: {optimizer_name.upper()}")
    global_step = 0
    metrics_path = f"metrics_{optimizer_name}.csv"
    with open(metrics_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        tf.keras.backend.set_value(optimizer.lr, combined_schedule(epoch))

        total_loss = 0.0
        step = 0
        progress_bar = tqdm(raw_ds, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for batch in progress_bar:
            aug1 = tf.reshape(batch["aug1"], [batch_size, 440, 129])
            aug2 = tf.reshape(batch["aug2"], [batch_size, 440, 129])


            with tf.GradientTape() as tape:
                zis = simclr_model(aug1, training=True)
                zjs = simclr_model(aug2, training=True)
                loss = criterion(zis, zjs)
            grads = tape.gradient(loss, simclr_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, simclr_model.trainable_variables))

            total_loss += loss
            step += 1
            global_step += 1
            avg_loss = total_loss / step
            current_lr = float(tf.keras.backend.get_value(optimizer.lr))
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.6f}"})
            del aug1, aug2, zis, zjs, grads, loss
            gc.collect()

        # 🔍 Validação
        print("🔎 Validação...")
        val_loss = 0.0
        val_steps = 0
        
        # Só calcula os embeddings se for época de salvar t-SNE
        save_tsne = (epoch + 1) % val_every == 0 or epoch == epochs - 1
        #val_embeds = [] if save_tsne else None
        
        for val_batch in val_ds:
            #orig = tf.reshape(val_batch["orig"], [batch_size, 440, 129])
            aug1 = tf.reshape(val_batch["aug1"], [batch_size, 440, 129])
            aug2 = tf.reshape(val_batch["aug2"], [batch_size, 440, 129])
            
            zis = simclr_model(aug1, training=False)
            zjs = simclr_model(aug2, training=False)
            loss_val = criterion(zis, zjs)
            
            val_loss += loss_val
            val_steps += 1
        
            # Apenas acumula embeddings se necessário
            #if save_tsne:
            #    embed = simclr_model(orig, training=False)
            #    val_embeds.append(embed)
        
        val_loss /= val_steps
        print(f"✅ Val Loss: {val_loss:.4f}")
        
        # 🔸 Salva métricas
        with open(metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, float(avg_loss), float(val_loss)])
        
        # 💾 Salva modelo + t-SNE somente se for necessário
        if save_tsne:
            print("💾 Salvando modelo e t-SNE...")
            encoder_model.save_weights(f"ckpt/encoder_{optimizer_name}_epoch{epoch+1}.h5")
            
            #val_embeds = np.concatenate(val_embeds, axis=0)
            #plot_tsne(val_embeds, epoch=epoch+1, save_path_prefix=f"val_tsne_{optimizer_name}")
            #print(f"🖼️ t-SNE salvo em 'val_tsne_{optimizer_name}_{epoch+1}.png'.")
        del aug1, aug2, zis, zjs, val_loss
        gc.collect()
    print("✅ Treinamento finalizado.")


if __name__ == "__main__":
    run_pretrain_simclr(optimizer_name="sgd")
