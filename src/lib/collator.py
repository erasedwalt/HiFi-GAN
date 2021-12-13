import math
from torch.nn.utils.rnn import pad_sequence

from typing import Tuple, Dict, List


FILL = math.log(1e-5)


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, melspec, melspec_loss = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)

        melspec = pad_sequence([
            melspec_[0].transpose(0, 1) for melspec_ in melspec
        ], padding_value=FILL).permute(1, 2, 0)

        melspec_loss = pad_sequence([
            melspec_[0].transpose(0, 1) for melspec_ in melspec_loss
        ], padding_value=FILL).permute(1, 2, 0)
        return waveform, melspec, melspec_loss
