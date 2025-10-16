from src.model.conformer_impl.conv_module import ConvolutionModule
from src.model.conformer_impl.ffn_module import FFNModule
from src.model.conformer_impl.mhsa_module import MHSAModule

import torch.nn as nn
import torch

# class ConformerEncoder(nn.Module):
#     def __init__(self):
#         super().__init__(self)
    
#     def forward(x):
#         return x

class EncoderLayer(nn.Module):
    def __init__(self, in_features, ffn_hidden_size, num_heads, depthwise_kernel_size, dropout = 0.0):
        super().__init__()
        self.conv = ConvolutionModule(in_features,
                                      hidden_size = in_features, # pytorch uses this
                                      depthwise_kernel_size = depthwise_kernel_size,
                                      dropout = dropout,
        )
        self.mhsa = MHSAModule(in_features, num_heads, dropout = dropout)
        self.ffn1 = FFNModule(in_features, ffn_hidden_size, dropout = dropout)
        self.ffn2 = FFNModule(in_features, ffn_hidden_size, dropout = dropout)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, input, attn_padding_mask = None):
        # TODO: pytorch for some reason does transpose(0, 1) before and after the convolution
        output = self.ffn1(input)
        output = self.mhsa(output, attn_padding_mask)
        output = self.conv(output)
        output = self.ffn2(output)
        output = self.layer_norm(output)
        return output

class ConformerEncoder(nn.Module):
    def __init__(self, in_features, ffn_hidden_size, num_heads, depthwise_kernel_size, num_layers, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(in_features, ffn_hidden_size, num_heads, depthwise_kernel_size, dropout = dropout) \
            for _ in range(num_layers)
        )

    def forward(self, input, lengths):
        batch_size = lengths.shape[0]
        max_time = lengths.max().item()
        padding_mask = torch.arange(max_time, device = lengths.device).expand(batch_size, max_time) \
            >= lengths.unsqueeze(1)
        output = input
        for l in self.layers:
            output = l(output, padding_mask)
        return output, lengths

def test_conformer_encoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 13
    n_timesteps = 100
    in_features = 119
    input = torch.randn(batch_size, n_timesteps, in_features).to(device)

    ffn_hidden_size = 179
    num_layers = 5
    depthwise_kernel_size = 25
    num_heads = 7
    model = ConformerEncoder(in_features, ffn_hidden_size, num_heads, depthwise_kernel_size, num_layers).to(device)

    output, _ = model(input)
    assert output.shape == input.shape
