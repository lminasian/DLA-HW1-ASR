import torch
import librosa


class TimeStretch:
    def __init__(self, speedup = 1.1, sample_rate = 16_000):
        self.speedup = speedup
        self.sr = sample_rate

    def __call__(self, wav):
        assert wav.shape[0] == 1
        return torch.tensor(
            librosa.effects.time_stretch(
                wav.numpy().squeeze(), sr = self.sample_rate, rate=self.speedup),
            device = wav.device,
        ).unsqueeze(0)


class SpeedPerturbation:
    def __init__(self, possible_speedups):
        self.possible_speedups = possible_speedups

    def __call__(self, wav):
        assert wav.shape[0] == 1
        speedup_idx = torch.randint(0, len(self.possible_speedups), (1,))[0].item()
        speedup = self.possible_speedups[speedup_idx]
        return torch.tensor(
            librosa.effects.time_stretch(wav.numpy().squeeze(), rate=speedup),
            device = wav.device,
        ).unsqueeze(0)