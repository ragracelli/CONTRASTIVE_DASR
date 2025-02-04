#TODO: Treinar o UPSTREAM com TORGO também para verificar a performance do modelo

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import string
from jiwer import cer, wer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import plot
import gc
import random

# Logging
'''
import logging

# Configura o logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Cria um manipulador de arquivo
file_handler = logging.FileHandler('pipeline.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Cria um manipulador de console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Adiciona ambos os manipuladores ao logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logging.getLogger().addHandler(console_handler)

# Log de diferentes níveis de severidade
logger.debug('Debug message')
logger.info('Information message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')
'''
# Configs
# Base paths for datasets
dysarthric_base_path = "/home/gracelli/databases/uaspeech/dysarthric/data_split/wav"
control_base_path = "/home/gracelli/databases/uaspeech/control/data_split/wav"
batch_size = 8
us_epochs = 30
ds_epochs = 100
sample_rate = 16000
# Create DataLoaders for % of the dataset
fraction = 1

# Section 1: Preprocessing class
class Preprocessor:
    def __init__(self, sample_rate=16000):
        print("Initializing Preprocessor...")
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


    def preprocess(self, audio_path):
        #print(f"Preprocessing audio: {audio_path}")
        waveform, sample_rate_orig = torchaudio.load(audio_path)
        if sample_rate_orig != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate_orig, new_freq=self.sample_rate)(waveform)
        #Normalization
        eps = 1e-8  
        waveform = (waveform - waveform.mean()) / (waveform.abs().max() + eps)        # Processar áudio com o Wav2Vec2Processor
        input_values = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").input_values
        return input_values

def plot_side_by_side(x1, x2, sample_rate, title_prefix=""):

    if x1.dim() == 3 and x1.size(1) == 1:
        x1 = x1.squeeze(1)
    if x2.dim() == 3 and x2.size(1) == 1:
        x2 = x2.squeeze(1)
    
    stft_transform = T.Spectrogram(n_fft=400, hop_length=200)
    amplitude_to_db = T.AmplitudeToDB()
    
    batch_size = x1.size(0)
    
    for i in range(batch_size):
        waveform1 = x1[i].detach().cpu().view(-1)  
        waveform2 = x2[i].detach().cpu().view(-1)  
        
        time_axis1 = torch.linspace(0, waveform1.shape[0] / sample_rate, steps=waveform1.shape[0]).numpy()
        time_axis2 = torch.linspace(0, waveform2.shape[0] / sample_rate, steps=waveform2.shape[0]).numpy()
        
        waveform1_for_spec = waveform1.unsqueeze(0)  # [1, T]
        waveform2_for_spec = waveform2.unsqueeze(0)  # [1, T]
        
        spec1 = stft_transform(waveform1_for_spec)      
        spec1_db = amplitude_to_db(spec1)               
        spec2 = stft_transform(waveform2_for_spec)
        spec2_db = amplitude_to_db(spec2)
        
        # channel removal dimension
        spec1_db_2d = spec1_db.squeeze(0)  # [freq_bins, time_frames]
        spec2_db_2d = spec2_db.squeeze(0)  # [freq_bins, time_frames]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        
        # Plot oscilogram x1
        axs[0, 0].plot(time_axis1, waveform1.numpy())
        axs[0, 0].set_title(f"{title_prefix} x1 Waveform (Exemplo {i+1})")
        axs[0, 0].set_xlabel("Tempo (s)")
        axs[0, 0].set_ylabel("Amplitude")
        
        # Plot oscilogram x2
        axs[0, 1].plot(time_axis2, waveform2.numpy())
        axs[0, 1].set_title(f"{title_prefix} x2 Waveform (Exemplo {i+1})")
        axs[0, 1].set_xlabel("Tempo (s)")
        axs[0, 1].set_ylabel("Amplitude")
        
        # Plot spectrogram x1
        im1 = axs[1, 0].imshow(spec1_db_2d.numpy(), origin="lower", aspect="auto",
                                extent=[0, waveform1.shape[0] / sample_rate, 0, sample_rate/2])
        axs[1, 0].set_title(f"{title_prefix} x1 STFT Spectrogram (Exemplo {i+1})")
        axs[1, 0].set_xlabel("Tempo (s)")
        axs[1, 0].set_ylabel("Frequência (Hz)")
        fig.colorbar(im1, ax=axs[1, 0])
        
        # Plot spectrogram x2
        im2 = axs[1, 1].imshow(spec2_db_2d.numpy(), origin="lower", aspect="auto",
                                extent=[0, waveform2.shape[0] / sample_rate, 0, sample_rate/2])
        axs[1, 1].set_title(f"{title_prefix} x2 STFT Spectrogram (Exemplo {i+1})")
        axs[1, 1].set_xlabel("Tempo (s)")
        axs[1, 1].set_ylabel("Frequência (Hz)")
        fig.colorbar(im2, ax=axs[1, 1])
        
        plt.tight_layout()
        plt.show()
        return x1, x2
    
# Section 2: Dataset class
class UASpeechDataset(Dataset):
    def __init__(self, base_path, split, preprocessor, fraction=1.0):
        print(f"Initializing UASpeechDataset for {split} split...")
        self.base_path = os.path.join(base_path, split)
        self.preprocessor = preprocessor
        # Collect all audio files
        all_files = [
            os.path.join(root, file)
            for root, _, files in tqdm(os.walk(self.base_path), desc=f"Loading {split} dataset")
            for file in files
            if file.endswith(".wav")
        ]
        # Select a fraction of the dataset (1% in this case)
        self.audio_files = all_files[:int(len(all_files) * fraction)]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Removed print to avoid excessive output
        audio_path = self.audio_files[idx]
        #waveform, _ = torchaudio.load(audio_path)
        waveform = self.preprocessor.preprocess(audio_path)
        return waveform, audio_path
    
preprocessor = Preprocessor(sample_rate=16000)

# Section 3: Function to create DataLoaders
#TODO: Partition bases proportion 62:31:7 - Bases already created by the Speech Transformer
def create_dataloaders(base_path, batch_size, collate_fn, fraction=1.0):
    print("Creating DataLoaders...")
    datasets = {
        "train": UASpeechDataset(base_path, "train", preprocessor, fraction),
        "dev": UASpeechDataset(base_path, "dev", preprocessor, fraction),
        "test": UASpeechDataset(base_path, "test",preprocessor,fraction),
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            drop_last=True,  # Ignore batches that are smaller than batch_size
            num_workers=4,
            pin_memory=True
        )
        for split in ["train", "dev", "test"]
    }

    return dataloaders


def collate_fn(batch):
    # Message moved to initialization stage
    waveforms, paths = zip(*batch)
    
    # Find the maximum length for padding
    max_len = max(waveform.size(-1) for waveform in waveforms)

    # Pad each tensor to match the maximum length
    #padded_waveforms = [waveform[:max_len] if waveform.size(-1) > max_len else torch.nn.functional.pad(waveform, (0, max_len - waveform.size(-1))) for waveform in waveforms]
    # **Normaliza e padroniza os tensores**
    padded_waveforms = [
        waveform[:max_len] if waveform.size(-1) > max_len else
        torch.nn.functional.pad(waveform, (0, max_len - waveform.size(-1)))
        for waveform in waveforms
    ]
    waveforms = torch.stack(padded_waveforms, dim=0)
    return waveforms, paths


dysarthric_dataloaders = create_dataloaders(
    dysarthric_base_path, batch_size, collate_fn, fraction
)
control_dataloaders = create_dataloaders(
    control_base_path, batch_size, collate_fn, fraction
)

# Display dataset sizes
print("\n# Section 4: Dataset Sizes")
for split in ["train", "dev", "test"]:
    print(f"[Dysarthric] {split}: {len(dysarthric_dataloaders[split].dataset)} audios")
    print(f"[Control] {split}: {len(control_dataloaders[split].dataset)} audios")

def create_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """
    Cria um otimizador Adam com weight decay (regularização L2)
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay  # Aplica o weight decay (L2 regularization)
    )
    return optimizer


def create_lr_scheduler(optimizer, patience=5, factor=0.5):

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  
        factor=factor,  
        patience=patience,  
        verbose=True,  
        min_lr=1e-6  
    )
    return scheduler


def apply_regularization(model, dropout_prob=0.5):

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  
            module.dropout = torch.nn.Dropout(p=dropout_prob)  
    
    return model

class Identity(nn.Module):
    def forward(self, x):
        return x

# Section 5: Define SimCLR model with HuBERT encoder
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True, use_bn=False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.use_bias and not self.use_bn)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(self.out_features)
            self.bn = nn.LayerNorm(self.out_features)
    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, head_type='nonlinear', **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, False, False)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, False, True))
        
    def forward(self, x):
        x = self.layers(x)
        return x

class SimCLR(nn.Module):
    def __init__(self, latent_dim=64, projector_hidden_dim=2048, projector_out_dim=128):
        super(SimCLR, self).__init__()

        # load model Wav2Vec2ForCTC
        self.pretrained = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.pretrained.lm_head = Identity()
        # freeze encoder layers
        for param in self.pretrained.parameters():
            param.requires_grad = False
            
        # number encoder layers to freeze
        num_layers_to_unfreeze = 1
        
        # iterates over last layers
        for layer in self.pretrained.wav2vec2.encoder.layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Simplifing model (optional)
        self.pretrained.wav2vec2.encoder.layers = torch.nn.ModuleList(
            self.pretrained.wav2vec2.encoder.layers[:11]
        )
        
        # Projection head for contrstive learning
        self.projector = ProjectionHead(in_features=768,
                                        hidden_features=projector_hidden_dim,
                                        out_features=projector_out_dim, head_type='nonlinear')
        
    def forward(self, x):
        x = x.squeeze(1)
        outputs = self.pretrained(x.squeeze(1))  # [batch_size, seq_len, feature_dim]
        logits = outputs.logits

        xp = self.projector(logits)
        
        return xp

class SimCLR_Loss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=-1)

    def mask_correlated_samples(self, batch_size):

        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: tensors with shape [B, T, F]
        """
        B, T, F = z_i.shape
        loss_total = 0.0

        for t in range(T):
            z_i_t = z_i[:, t, :]  # [B, F]
            z_j_t = z_j[:, t, :]  # [B, F]

            z = torch.cat((z_i_t, z_j_t), dim=0)
            N = 2 * B
            mask = self.mask_correlated_samples(B).to(z.device)

            sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

            sim_i_j = torch.diag(sim, B)  # must be size B
            sim_j_i = torch.diag(sim, -B) # must be size B

            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
            negative_samples = sim[mask].view(N, -1)

            labels = torch.zeros(N, device=z.device, dtype=torch.long)
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss_t = self.criterion(logits, labels)
            loss_t /= N
            loss_total += loss_t

        return loss_total / T

# Section 6: Augmentation functions
# TODO usar uma rede attention transformer pre-treinada para localizar os pontos
# de maior energia no espectro para aplicar a mascara  
def segment_audio_raw_tensor(
    waveform: torch.Tensor,
    sr: int,
    silence_threshold: float = 0.001,
    silence_ratio_range: tuple = (0.02, 0.08),
    max_silence_points: int = 2,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Identifies the active region (non-silent) of the signal and, within it, randomly inserts
    up to two silent segments. The start point and duration of the silence are defined randomly.

    Parameters:
      - waveform (torch.Tensor): Audio tensor with shape [channels, time].
      - sr (int): Sampling rate.
      - silence_threshold (float): Threshold for considering a sample as "active." Values below
                                   this threshold are considered silence.
      - silence_ratio_range (tuple): Relative range (minimum, maximum) for defining the size of the silence
                                     in relation to the active region's duration.
      - max_silence_points (int): Maximum number of silent segments to be inserted (1 or 2).
      - device (torch.device): Device for processing (GPU or CPU).

    Returns:
      - torch.Tensor: Modified audio tensor with silence inserted into the active region.
    """

    # Move o tensor para o dispositivo desejado
    waveform = waveform.to(device)
    
    # stereo ---> mono
    if waveform.shape[0] > 1:
        waveform_mono = waveform[0:1, :]
    else:
        waveform_mono = waveform
        
    # uses first channel (tensor 1D)
    signal = waveform_mono[0]
    
    # Identifica os índices onde o sinal está "ativo" (acima do limiar)
    # Usando flatten() para garantir que teremos uma lista de inteiros
    silence_threshold = torch.abs(signal)
    '''
    # Stats for choose threshold
    mean_value = silence_threshold.mean()
    min_value = silence_threshold.min()
    max_value = silence_threshold.max()
    std_value = silence_threshold.std()
    var_value = silence_threshold.var()
    sum_value = silence_threshold.sum()
    median_value = silence_threshold.median()

    # Imprimindo as estatísticas
    print(f'Média: {mean_value.item()}')
    print(f'Mínimo: {min_value.item()}')
    print(f'Máximo: {max_value.item()}')
    print(f'Desvio Padrão: {std_value.item()}')
    print(f'Variância: {var_value.item()}')
    print(f'Soma: {sum_value.item()}')
    print(f'Mediana: {median_value.item()}')
    power_max = torch.max(signal**2)
    rms_value = torch.sqrt(torch.mean(signal**2))
    #print(power_max)
    #print(rms_value)
    '''
    silence_threshold = silence_threshold.max()/(2**(0.5))
    print(silence_threshold)
    active_indices = torch.nonzero((torch.abs(signal)) > silence_threshold).flatten().tolist()
    #mask = active_indices!=0
    #active_indices = active_indices[mask]
    print(active_indices)
    # Se não houver pontos ativos, retorne o sinal original
    if len(active_indices) == 0:
        print("Warning: Nenhuma região ativa encontrada. Retornando sinal original.")
        return waveform
    
    # Define a região ativa: do primeiro ao último índice ativo
    region_start = active_indices[0]
    region_end = active_indices[-1]
    region_length = region_end - region_start
    if region_length <= 0:
        print("Warning: Região ativa muito curta. Retornando sinal original.")
        return waveform
    
    # Decide aleatoriamente quantos pontos de silêncio inserir: 1 ou 2 (mas não mais que o definido)
    num_silence_points = random.randint(1, max_silence_points)
    
    for _ in range(num_silence_points):
        # Define aleatoriamente a duração do silêncio como uma fração da região ativa
        min_silence = int(silence_ratio_range[0] * region_length)
        max_silence = int(silence_ratio_range[1] * region_length)
        if max_silence < min_silence:
            max_silence = min_silence
        silence_length = random.randint(min_silence, max_silence)
        #print(silence_length)
        # Escolhe aleatoriamente um ponto de início para o silêncio, garantindo que o trecho caiba dentro da região
        if region_length - silence_length > 0:
            silence_start = random.randint(region_start, region_end - silence_length)
        else:
            silence_start = region_start
        silence_end = silence_start + silence_length
        #print(silence_start, silence_end)
        # Insere o silêncio (substitui por zero)
        signal[...,silence_start:silence_end] = 0 #TODO: Somente para visualização. Podemos mudar para valores inteiros negativos ou positivos para evidenciar melhor na imagem. 
    # Atualiza o canal modificado
    waveform_mono[0] = signal

    # Se o áudio original for multi-canal, atualiza apenas o primeiro canal
    if waveform.shape[0] > 1:
        waveform[0, :] = waveform_mono[0]
        return waveform
    else:
        return waveform_mono

'''
def add_random_silence(audio, max_silence=0.1):
    # Message moved to initialization stage
    energy = torch.quantile(audio.pow(2).mean(dim=-1), 0.75)
    silence_length = int(max_silence * audio.size(-1))
    start_idx = max(0, torch.argmax(energy) - silence_length // 2)
    end_idx = min(audio.size(-1), start_idx + silence_length)

    audio[..., start_idx:end_idx] = 0
    return audio
'''
def add_white_noise(audio, noise_level=0.1):
    noise = torch.randn_like(audio) * noise_level
    return (audio + noise).clamp(-1, 1)  # 🔥 Evita estouro de valores
'''
def apply_time_stretch(audio, rate=0.9):
    #print(f"Applying time stretch with rate: {rate}...")
    device = audio.device
    transform = TimeStretch(n_freq=audio.size(1), fixed_rate=rate).to(device)
    stretched = transform(audio)
    # Converte o tensor complexo para real usando .real
    return stretched.real
'''
def apply_time_stretch(audio: torch.Tensor, sample_rate: int, factor: float = 0.9) -> torch.Tensor:
    """
    Aplica time stretch via reamostragem ao áudio e ajusta seu tamanho para ser igual ao original.

    Parâmetros:
      - audio (torch.Tensor): Tensor contendo os dados de áudio. Pode ser mono ([n_amostras])
                                ou multi-canais ([canais, n_amostras]). Deve estar no dispositivo correto.
      - sample_rate (int): Taxa de amostragem original do áudio.
      - factor (float): Fator de time stretch. Valores menores que 1 deixam o áudio mais rápido (mais curto),
                        enquanto valores maiores que 1 deixam o áudio mais lento (mais longo).

    Retorna:
      - torch.Tensor: Áudio com time stretch aplicado e tamanho ajustado para ser igual ao original.
    """
    # Calcula a nova taxa de amostragem baseada no fator
    new_sample_rate = int(sample_rate * factor)

    # Cria o objeto resampler e move-o para o dispositivo
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate).to(device)

    # Aplica o resampler ao áudio
    stretched_audio = resampler(audio)

    # Ajusta o tamanho do áudio para ser igual ao original
    original_length = audio.shape[-1]
    stretched_length = stretched_audio.shape[-1]

    if stretched_length > original_length:
        # Se o áudio transformado estiver maior, faz um crop (corta) no início
        stretched_audio = stretched_audio[..., :original_length]
    elif stretched_length < original_length:
        # Se estiver menor, preenche com zeros no final
        pad_amount = original_length - stretched_length
        stretched_audio = F.pad(stretched_audio, (0, pad_amount))

    return stretched_audio


# Section 7: Training loop for upstream (SimCLR)

#Dimesity reduction
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = []
    indices = []
    min_length = float('inf')  # Inicializa como infinito

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch[0].to(device)
            output = model(inputs)
            min_length = min(min_length, output.size(1))  # Atualiza para o menor tamanho
            embeddings.append(output.detach().cpu())
            indices += [i] * inputs.size(0)

    # Truncar todos os embeddings para o menor tamanho encontrado
    truncated_embeddings = [emb[:, :min_length] for emb in embeddings]

    embeddings = torch.cat(truncated_embeddings)
    indices = torch.tensor(indices)

    return embeddings, indices



def plot_tsne_3d(embeddings, indices, epoch, reverse_vocab):
    print("Plotting t-SNE")

    # Verificar a forma dos embeddings
    print("Shape of embeddings before reshape:", embeddings.shape)

    # Se os embeddings tiverem três dimensões, achate as duas últimas dimensões
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        print("Shape of embeddings after reshape:", embeddings.shape)

    # Garantir que os índices sejam mapeados para 32 classes
    # Normalização dos índices para garantir que fiquem entre 0 e 31
    indices = indices % 32  # Isso garante que os valores de índice estejam entre 0 e 31

    # Agora, aplicar o t-SNE
    from sklearn.manifold import TSNE
    import plotly.graph_objects as go

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Gerar as letras correspondentes aos índices
    # Use o vocab para mapear os índices para as letras
    labels = [reverse_vocab[i] for i in indices]  # Mapear os índices para as letras

    # Plotting
    fig = go.Figure(data=[go.Scatter3d(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        z=tsne_results[:, 2],
        mode='markers+text',  # Adicionar texto (as letras) nos pontos
        marker=dict(
            size=5,
            color=indices,  # Usar os índices para colorir os pontos
            colorscale='Jet',  # Escolher uma escala de cores adequada
            colorbar=dict(title="Class"),  # Barra de cores para indicar as classes
            opacity=0.8
        ),
        text=labels,  # As letras que representam cada classe
        textposition='top center'  # Posicionar o texto acima dos pontos
    )])

    # Atualizar layout do gráfico
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), title="3D t-SNE Visualization")

    # Salvar a figura como um arquivo HTML interativo
    fig.write_html(f'3d-tsne_{epoch}.html')
    plot(fig, filename='3D_t-SNE_Visualization.html')
    print("Interactive plot saved in 3D_t-SNE_Visualization.html")


def train_simclr(model, dataloader, optimizer, criterion, device, epochs, sample_rate, checkpoint_path, val_dataloader=None):
    print("Starting SimCLR upstream training...")

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        count = 0
        batches_per_step = 4  # Number of batches to process before clearing memory

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} Training Batches", leave=True), 1):
            x, _ = batch
            x = x.to(device)  # Move the batch to the specified device (GPU or CPU)
            #TODO Coeficiente das transformações obtidas de forma empírica, 
            #para evitar saturação e outras distorções
            
            # Apply transformations
            x_orig = x.clone()
            # Example placeholders for additional transformations
            # x_white = add_white_noise(x, noise_level=0.1)
            # x_stretch = apply_time_stretch(x, sample_rate, factor=0.9)
            # x_occlusion = segment_audio_raw_tensor(x, sample_rate, device=device)
            # min_length = min(x_white.size(-1), x_stretch.size(-1), x_occlusion.size(-1))
            # x1 = x_white[..., :min_length] + x_stretch[..., :min_length]
            x2 = segment_audio_raw_tensor(x * 0.25, sample_rate, device=device)
            x2 = apply_time_stretch(x2, sample_rate, factor=1.1)
            x2 = add_white_noise(x2 * 0.25, noise_level=0.01)

            # Display original and transformed tensors for the first few batches
            if count < 8:
                plot_side_by_side(x_orig, x2, sample_rate, title_prefix="Compare:")
                count += 1
            
            # Perform forward pass for original and transformed inputs
            z_i = model(x)
            z_j = model(x2)

            # Compute loss
            raw_loss = criterion(z_i, z_j)
            loss = raw_loss / batches_per_step  # Normalize loss by the number of accumulated batches
            loss.backward()  # Perform backpropagation

            # Perform optimization step and zero the gradients
            if i % batches_per_step == 0 or i == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += raw_loss.item()

            # Memory cleanup to prevent CUDA out of memory errors
            if i % batches_per_step == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Periodically print the current batch loss
            if i % 10 == 0:
                tqdm.write(f"[Batch {i}] Current Batch Loss: {raw_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save a checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_filename = f"{checkpoint_path}/simclr_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")
        
        # Extract e plot embeddings each 10 epochs
        if (epoch + 1) % 3 == 0:
            embeddings, indices = extract_embeddings(control_dataloaders['dev'], simclr_model, device)
            embeddings_np = embeddings.detach().cpu().numpy()
            indices_np = indices.detach().cpu().numpy() if indices is not None else None
            plot_tsne_3d(embeddings_np, indices_np, epoch, reverse_vocab)


def validate_simclr(model, dataloader, criterion, device):
    print("Starting validation...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation Batches", leave=True)):
            x, _ = batch
            x = x.to(device)

            # Data augmentations
            x_orig = x.clone()
            x_white = add_white_noise(x, noise_level=0.1)
            x_stretch = apply_time_stretch(x, sample_rate, factor=0.9)
            x_occlusion = segment_audio_raw_tensor(x, sample_rate, device=device)
            
            # Determina o comprimento mínimo entre os dois resultados
            min_length = min(x_white.size(-1), x_stretch.size(-1), x_occlusion.size(-1))
                             
            # Recorta ambos os tensores para o mesmo tamanho e os soma
            #x1 = x_white[..., :min_length] + x_stretch[..., :min_length]
            x2 = x_white[..., :min_length] + x_stretch[..., :min_length] + x_occlusion[..., :min_length]
            #x2 = x_occlusion


            # SimCLR forward
            z_i = model(x_orig)
            z_j = model(x2)

            loss = criterion(z_i, z_j)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def plot_curves(loss_train_list, loss_val_list, train_acc_list, val_acc_list, loss_or_acc):
    """
    Plota as curvas de perda de treinamento e validação ao longo das épocas.

    Parâmetros:
      loss_train_list: Lista de valores de perda de treinamento.
      loss_val_list: Lista de valores de perda de validação.
    """
    # Número de épocas baseado no tamanho da lista de perda
    if loss_or_acc == 'loss':
        label_train = 'Train Loss'
        label_valid = 'Validation_Loss'
        title = 'Loss curves of Train and Validation'
        train = loss_train_list
        val = loss_val_list
    else:
        label_train = 'Train Accuracy'
        label_valid = 'Validation Accuracy'
        title = 'Accuracy curves of Train and Validation'
        train = train_acc_list
        val = val_acc_list
    epochs = range(1, len(train) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plota a perda de treinamento
    plt.plot(epochs, train, marker='o', label=label_train)
    
    # Plota a perda de validação
    plt.plot(epochs, val, marker='x', label=label_valid)
    
    # Configura os eixos e o título
    plt.xlabel('Epoch')
    plt.ylabel(loss_or_acc)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.show()


# Section 8: Creating Dataset for downstream

def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # Para evitar divisão por zero, caso o tensor tenha valores constantes
    if tensor_max == tensor_min:
        return tensor  # Retorna o tensor original se os valores forem constantes

    # Normalizando para o intervalo [-1, 1]
    return 2 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1

class UASpeechDatasetDownstream(Dataset):
    def __init__(self, base_path, split, transcription_map, preprocessor, char_to_idx, fraction=1.0):
        print(f"Initializing UASpeechDataset for {split} split...")
        self.base_path = os.path.join(base_path, split)
        self.preprocessor = preprocessor

        all_files = [
            os.path.join(root, file)
            for root, _, files in tqdm(os.walk(self.base_path), desc=f"Loading {split} dataset")
            for file in files
            if file.endswith(".wav")
        ]
        self.audio_files = all_files[:int(len(all_files) * fraction)]  # Limita a fração
        self.transcription_map = transcription_map
        self.char_to_idx = char_to_idx
        

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Carrega o áudio
        audio_path = self.audio_files[idx]
        #waveform, _ = torchaudio.load(audio_path)
        waveform = self.preprocessor.preprocess(audio_path)

        #waveform = normalize_tensor(waveform)
        #print(f"Intervalo de x após normalização: Mínimo: {waveform.min().item()}, Máximo: {waveform.max().item()}", flush=True)
        
        # Extrai a chave do nome do arquivo
        file_name = os.path.basename(audio_path)
        parts = file_name.split("_")
        
        # Extraímos a parte da chave
        key_part = parts[2]  # Exemplo: 'UW1', 'UW2', etc.
        
        # Se a chave começar com 'UW', adiciona a parte à esquerda (B1, B2, etc.)
        if key_part.startswith("UW"):
            left_part = parts[1]  # Exemplo: 'B1', 'B2', etc.
            key = left_part + "_" + key_part  # Combina a parte à esquerda com a parte da chave
        else:
            key = key_part  # Se não começar com 'UW', usa a chave como está

        # 🔹 Mapeia a chave para a transcrição
        transcription = self.transcription_map.get(key, "")
        transcription = transcription.upper()
        #print(transcription)

        # 🔹 Se a transcrição estiver vazia, usar um valor padrão
        if not transcription:
            transcription = "<pad>"

        # 🔹 Converte a transcrição para índices usando `char_to_idx` do modelo Wav2Vec2
        transcription_indices = [
            char_to_idx.get(char, char_to_idx.get('<unk>', char_to_idx.get(' ', 0))) 
            for char in transcription
        ]

        ## 🔹 Preenche a transcrição com `<pad>` até atingir o comprimento máximo
        #transcription_indices = transcription_indices[:self.max_len]  # Trunca se maior
        #transcription_indices += [char_to_idx.get('<pad>', 0)] * (self.max_len - len(transcription_indices))  # Padding
        #print(transcription_indices)
        # 🔹 Retorna o áudio e a transcrição como tensor
        return waveform, torch.tensor(transcription_indices)

# Função para criar os DataLoaders
def create_dataloaders_downstream(base_path, batch_size, collate_fn, transcription_map, char_to_idx, fraction=1.0):
    print("Creating DataLoaders for Downstream...")
    datasets = {
        "train": UASpeechDatasetDownstream(base_path, "train", transcription_map, preprocessor, char_to_idx, fraction),
        "dev": UASpeechDatasetDownstream(base_path, "dev", transcription_map, preprocessor, char_to_idx, fraction),
        "test": UASpeechDatasetDownstream(base_path, "test", transcription_map, preprocessor, char_to_idx, fraction),
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            drop_last=True,   # Ignore batches that are smaller than batch_size
            num_workers=4,
            pin_memory=True
        )
        for split in ["train", "dev", "test"]
    }

    return dataloaders

# Create character mapping
def create_char_mapping(transcriptions):
    # Inclui letras, números e caracteres especiais como espaço e pontuação
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + string.whitespace
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # Índice começando de 1
    char_to_idx['<pad>'] = 0  # Adicionando um índice para o padding (opcional)
    
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

# Carregar o arquivo CSV de transcrições
dictionary_file = "dictionary_UASPEECH.csv"
transcription_dict = pd.read_csv(dictionary_file)

# Criar um dicionário com as chaves associadas às transcrições
transcription_map = dict(zip(transcription_dict['FILE NAME'], transcription_dict['WORD']))

# Criar o mapeamento de caracteres
#char_to_idx, idx_to_char = create_char_mapping(transcription_map.values())

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
char_to_idx = processor.tokenizer.get_vocab()
#print(char_to_idx)
idx_to_char= {idx: char for char, idx in char_to_idx.items()}
#print(idx_to_char)
vocab = char_to_idx
print(vocab)
#num_classes = vocab  # Número de classes no seu vocabulário, ajustado conforme necessário

# Load Downstream model
class DownstreamModel(nn.Module):
    def __init__(self, pretrained, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(DownstreamModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.pretrained = pretrained.to(self.device)  # Move o modelo pré-treinado para a GPU

        self.classifier = nn.Linear(128, num_classes).to(self.device)        
        
        for param in self.pretrained.parameters():
            param.requires_grad = False  # Congela os pesos do modelo pré-treinado
        
        for name, param in self.pretrained.named_parameters():
            if "encoder.layers.9" in name or "encoder.layers.10" in name or "encoder.layers.11" in name:
                param.requires_grad = True  # Ajusta apenas as camadas finais
            else:
                param.requires_grad = False        
                
        #self.classifier = nn.Linear(128, num_classes)  # Ajuste se necessário
            
    def forward(self, x):
        x = x.to(self.device)
        x = self.pretrained(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, feature_dim = x.shape

  
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        return x


# Section 9: Training loop for downstream (ASR)
def train_downstream(model, dataloader, optimizer, scheduler, vocab, device):
    print("Starting downstream training...")
    model.train()
    total_loss = 0
    total_correct = 0 
    total_samples = 0
    print("\n[Downstream Training: ASR]")
    # Lists to store training losses and metrics for plotting
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean')
    
    progress_bar = tqdm(dataloader, desc="Training Batches")
    
    for batch in progress_bar:
        # VVerifies if batch is a dictionary or tuple
        if isinstance(batch, dict):  # if dictionary, correctly extract
            x = batch["input"].to(device)
            y = batch["label"]
        else: 
            x, y = batch
            x = x.to(device)
            #y = [tensor.to(device) for tensor in y]

        logits = model(x)
        #N, T, C = logits.shape 
        predicted_ids = torch.argmax(logits, dim=-1)
        targets = torch.cat(y).to(device)
        batch_size = logits.shape[0]  # Get the size of batch
        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long).to(device)
        target_lengths = torch.tensor([len(x) for x in y], dtype=torch.long).to(device)
        logits_permute=logits.permute(1, 0, 2)
        
        loss = ctc_loss_fn(
            logits_permute,  
            targets,                     
            input_lengths,               
            target_lengths               
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()

        # Accuracy computation by aligning indices
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # Get predicted indices (highest probability)
            for i in range(batch_size):
                pred_seq = preds[i, :target_lengths[i]]  # Select only valid indices
                target_seq = targets[:target_lengths[i]]  # Select corresponding target indices

                correct_preds = (pred_seq == target_seq).sum().item()  # Count correct predictions
                total_correct += correct_preds
                total_samples += len(target_seq)  # Count valid target elements
        
        acc = correct_preds/total_samples
        progress_bar.set_postfix(loss=loss.item(), accuracy=acc)

    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch completed. Average Loss: {avg_loss:.4f}, Average Acc: {avg_accuracy:.4f}")
    # Append the average loss of this epoch to the loss list
    train_loss_list.append(avg_loss)
    train_acc_list.append(avg_accuracy)
    
    return avg_loss, avg_accuracy, train_loss_list, train_acc_list

# Section 10: Validation
def validate_downstream(model, dataloader, vocab, device):
    print("Starting validation...")
    model.eval()
    total_loss = 0
    total_correct = 0 
    total_samples = 0
    ctc_loss_fn = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, reduction='mean')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Batches"):
        # Verifies if dictionary or tuple
            if isinstance(batch, dict):  #If dictionary extract
                x = batch["input"].to(device)
                y = batch["label"]
            else:  
                x, y = batch
                x = x.to(device)
                #y = [tensor.to(device) for tensor in y]
    
            # Obtém logits do modelo
            logits = model(x)  # [batch_size, seq_len, feature_dim]
            predicted_ids = torch.argmax(logits, dim=-1)
            # Converts IDs to text
            #decoded_preds = [processor.decode(pred) for pred in predicted_ids[2]]
            #decoded_preds = processor.decode(predicted_ids[2])
            decoded_preds = [processor.decode(predicted_ids[i]) for i in range(predicted_ids.shape[0])]
            decoded_labels = decode_labels(y, reverse_vocab)
            print(f"Transcriptions Pred: {decoded_preds}")
            print(f"Supervised Labels: {decoded_labels}")
            #break
            targets = torch.cat(y).to(device)
 
            batch_size = logits.shape[0] 
            input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(x) for x in y], dtype=torch.long).to(device)
            logits_permute=logits.permute(1, 0, 2)

            
            loss = ctc_loss_fn(
                logits_permute,          
                targets,                     
                input_lengths,               
                target_lengths               
            )
            total_loss += loss.item()
            # Accuracy computation by aligning indices
            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # Get predicted indices (highest probability)
                for i in range(batch_size):
                    pred_seq = preds[i, :target_lengths[i]]  # Select only valid indices
                    target_seq = targets[:target_lengths[i]]  # Select corresponding target indices

                    correct_preds = (pred_seq == target_seq).sum().item()  # Count correct predictions
                    total_correct += correct_preds
                    total_samples += len(target_seq)  # Count valid target elements
            
            acc = correct_preds/total_samples
            #progress_bar.set_postfix(loss=loss.item(), accuracy=acc)

        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0

        val_loss = total_loss / len(dataloader)
        
        
        print(f"Validation completed. Average Loss: {val_loss:.4f}, Average Acc: {avg_accuracy:.4f}")
        val_wer, val_cer = calculate_metrics(decoded_preds, decoded_labels)

        print(f"Average WER: {val_wer:.4f}")
        print(f"Average CER: {val_cer:.4f}")

        val_loss_list.append(val_loss)
        val_cer_list.append(val_cer)
        val_wer_list.append(val_wer)
        val_acc_list.append(avg_accuracy)


    return val_loss, val_wer, val_cer, val_loss_list, val_cer_list, val_wer_list

def calculate_metrics(decoded_preds, decoded_labels):
    total_wer = 0
    total_cer = 0
    num_samples = len(decoded_preds)
    
    for pred, label in zip(decoded_preds, decoded_labels):
        total_wer += wer(label, pred)
        total_cer += cer(label, pred)

    average_wer = total_wer / num_samples
    average_cer = total_cer / num_samples

    return average_wer, average_cer


def create_reverse_vocab(vocab):
    return {idx: char for char, idx in vocab.items()}

reverse_vocab = create_reverse_vocab(vocab)

def decode_labels(tensor_list, vocab):
    all_decoded = []
    for tensor in tensor_list:
        decoded_chars = []
        for idx in tensor:
            if idx.item() != 0:
                decoded_chars.append(reverse_vocab.get(idx.item(), ''))
            elif decoded_chars:  
                all_decoded.append(''.join(decoded_chars))
                decoded_chars = [] 


        if decoded_chars:
            all_decoded.append(''.join(decoded_chars))

    return all_decoded


# Section 11: Main pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upstream training
simclr_model = SimCLR(latent_dim=128).to(device)
print(simclr_model)
#embeddings, indices = extract_embeddings(control_dataloaders['dev'], simclr_model, device)
#embeddings_np = embeddings.detach().cpu().numpy()
#indices_np = indices.detach().cpu().numpy() if indices is not None else None
#plot_tsne_3d(embeddings_np, indices_np, epochs=None)


optimizer = optim.Adam(simclr_model.parameters(), lr=1e-4)
#optimizer = create_optimizer(simclr_model, lr=1e-3, weight_decay=1e-4)
scheduler = create_lr_scheduler(optimizer)


def calculate_accuracy_and_confusion_matrix(preds, labels):
    # Convert predictions and labels to CPU arrays for calculation
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    accuracy = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)

    return accuracy, conf_matrix

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap='Blues', alpha=0.7)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    plt.xticks(range(len(conf_matrix)), range(len(conf_matrix)))
    plt.yticks(range(len(conf_matrix)), range(len(conf_matrix)))

    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')

    plt.show()


# Upstream Train
criterion = SimCLR_Loss(temperature=0.1)
train_loss_list = []
val_loss_list = []
for epoch in range(us_epochs):
    print(f"\nEpoch {epoch+1} of Upstream Training")
    train_loss = train_simclr(simclr_model, control_dataloaders["train"], optimizer, criterion, device)
    train_loss_list.append(train_loss)
    print(f"[Upstream] Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    val_loss = validate_simclr(simclr_model, control_dataloaders["dev"], criterion, device)
    val_loss_list.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}")
    
# Save Model
torch.save(simclr_model.state_dict(), 'up_temp0.1_10ep_all-aug.pth')
print("Model Weights succesul saved!")

# Plot curves train performance
plot_curves(train_loss_list, val_loss_list, train_acc_list=None, val_acc_list=None, loss_or_acc=None)


# Downstream training with CTC loss
#print(vocab)
num_classes = len(char_to_idx) 
asr_model = DownstreamModel(num_classes=num_classes, pretrained=simclr_model).to(device)
print(asr_model)
embeddings, indices = extract_embeddings(control_dataloaders['dev'], simclr_model, device)
embeddings_np = embeddings.detach().cpu().numpy()
indices_np = indices.detach().cpu().numpy() if indices is not None else None
plot_tsne_3d(embeddings_np, indices_np, epoch=None, reverse_vocab=reverse_vocab)

#for param in asr_model.parameters():
#    param.requires_grad = True  # Descongela todas as camadas
    
#optimizer = optim.Adam(asr_model.parameters(), lr=1e-4)
optimizer = create_optimizer(asr_model, lr=1e-3, weight_decay=1e-5)
#scheduler = create_lr_scheduler(optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# Criação dos dataloaders para o downstream
dysarthric_dataloaders_downstream = create_dataloaders_downstream(
    dysarthric_base_path, batch_size, collate_fn, transcription_map, char_to_idx, fraction=fraction
)

# Downstream train
best_val_loss = float('inf')
train_loss_list = []
train_acc_list =[]
val_acc_list = []
val_loss_list = []
val_cer_list = []
val_wer_list = []
for epoch in range(ds_epochs):
    print(f"\nEpoch {epoch+1} of Downstream Training")
    train_loss, train_acc, train_loss_list, train_acc_list = train_downstream(asr_model, dysarthric_dataloaders_downstream["train"], optimizer, scheduler, vocab, device)
    print(f"[Downstream] Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Validação
    #val_loss, val_wer, val_cer = validate(asr_model, dysarthric_dataloaders_downstream["dev"], vocab, max_len, device)
    val_loss, val_wer, val_cer, val_loss_list, val_cer_list, val_wer_list = validate_downstream(asr_model, dysarthric_dataloaders_downstream["dev"], vocab, device)
    print(f"[Downstream] Validation Loss: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
    #print(f"[Downstream] Validation Loss: {val_loss:.4f}")

    # Directory to save the model and the filename
    save_dir = 'models'
    save_filename = 'ds_wav2vec_sc-tmax10__lr1e-4_ep100.pth'

    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")

    # Full path for the file where the model will be saved
    save_path = os.path.join(save_dir, save_filename)

    # Check if the validation loss improved, and if so, save the model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"Validation loss improved. Saving model...")
        torch.save(asr_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# Plot the loss curves

plot_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list, loss_or_acc='loss')
plot_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list, loss_or_acc='accuracy')
# Testing
print("\n[Testing Phase]")
#test_loss, test_wer, test_cer = validate(asr_model, dysarthric_dataloaders["test"], criterion, device)
#print(f"[Test] Loss: {test_loss:.4f}, WER: {test_wer:.4f}, CER: {test_cer:.4f}")

'''
# Calculate Accuracy and Confusion Matrix
accuracy, conf_matrix = calculate_accuracy_and_confusion_matrix(all_preds, all_labels)
print(f"Accuracy: {accuracy:.4f}")
plot_confusion_matrix(conf_matrix)
'''