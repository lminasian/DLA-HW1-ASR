import torch
import librosa


class PitchShift:
    def __init__(self, n_steps, sr = 16_000):
        self.n_steps = n_steps
        self.sr = sr

    def __call__(self, wav):
        assert wav.shape[0] == 1
        return torch.tensor(
            librosa.effects.pitch_shift(
                wav.numpy().squeeze(), sr=self.sr, n_steps = self.n_steps)
        ).unsqueeze(0)