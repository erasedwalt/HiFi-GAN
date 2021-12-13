import torch
import torchaudio
import random
import pandas as pd


class LJSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, root, melspec, melspec_loss, chunk_size=8192, train=True):
        super().__init__()

        self.data = pd.read_csv(root)
        self.chunk_size = chunk_size
        self.melspec_computer = melspec
        self.melspec_loss_computer = melspec_loss
        self.train = train
        self.path = '/'.join(root.split('/')[:-1]) + '/wavs/'
    
    def __len__(self):
      return self.data.shape[0]

    def __getitem__(self, index: int):
        name = self.data.loc[index, '0'] + '.wav'
        waveform, sr = torchaudio.load(self.path + name)
        if self.train:
            if waveform.shape[-1] > self.chunk_size:
                rand_pivot = random.randint(0, waveform.shape[-1] - self.chunk_size - 1)
                waveform = waveform[:, rand_pivot:rand_pivot + self.chunk_size]
        melspec = self.melspec_computer(waveform)
        melspec_loss = self.melspec_loss_computer(waveform)
        return waveform, melspec, melspec_loss
