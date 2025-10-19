import torch


class FrequencyMasking:
    def __init__(self, max_masked_percentage):
        assert 0 <= max_masked_percentage <= 1
        self.max_masked_percentage = max_masked_percentage

    def __call__(self, spectrogram):
        spectrogram = spectrogram.clone()
        _, _, frequency_range  = spectrogram.shape
        max_masked_range = int(self.max_masked_percentage * frequency_range)
        width = torch.randint(0, max_masked_range, (1,))[0]
        f = torch.randint(0, frequency_range, (1,))[0]
        spectrogram[:, :, f: f + width] = 0
        return spectrogram


class TimeMasking:
    def __init__(self, max_masked_percentage):
        assert 0 <= max_masked_percentage <= 1
        self.max_masked_percentage = max_masked_percentage

    def __call__(self, spectrogram):
        spectrogram = spectrogram.clone()
        _, time_range, _  = spectrogram.shape
        max_masked_range = int(self.max_masked_percentage * time_range)
        width = torch.randint(0, max_masked_range, (1,))[0]
        t = torch.randint(0, time_range, (1,))[0]
        spectrogram[:, t:t + width, :] = 0
        return spectrogram
    

class SpecAug:
    def __init__(self, max_time_masking, max_frequency_masking):
        self.time_masking = TimeMasking(max_time_masking)
        self.frequency_masking = FrequencyMasking(max_frequency_masking)
    def __call__(self, spectrogram):
        output = spectrogram
        output = self.time_masking(output)
        output = self.frequency_masking(output)
        return output