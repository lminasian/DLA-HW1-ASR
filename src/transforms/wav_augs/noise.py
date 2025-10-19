import torch


class AddNoise:
    def __init__(self, var = 0.01):
        self.var = var

    def __call__(self, wav):
        noise = torch.rand_like(wav) * self.var
        result = torch.clip(wav + noise, -1, 1)
        return result