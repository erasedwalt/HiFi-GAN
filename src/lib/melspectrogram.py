import librosa
import torch
from torch import nn
import torchaudio

from typing import Dict, Union


class MelSpectrogram(nn.Module):

    def __init__(self, config: Dict[str, Union[int, float]], for_loss=False):
        super(MelSpectrogram, self).__init__()

        self.config = config
        self.pad_size = (config['n_fft'] - config['hop_length']) // 2

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(**config)

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = 1.0

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config['sample_rate'],
            n_fft=config['n_fft'],
            n_mels=config['n_mels'],
            fmin=config['f_min'],
            fmax=None if for_loss else config['f_max']
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        # for right spec length
        audio = torch.nn.functional.pad(
            audio.unsqueeze(1),
            (self.pad_size, self.pad_size),
            mode='reflect'
        )
        audio = audio.squeeze(1)

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
