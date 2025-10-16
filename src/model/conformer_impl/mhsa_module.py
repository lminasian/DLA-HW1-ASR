import torch
import torch.nn as nn

class MHSAModule(nn.Module):
    def __init__(self, in_features, num_heads, dropout = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.mhsa = nn.MultiheadAttention(in_features, num_heads,
                                          dropout = dropout, batch_first = True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, attn_padding_mask):
        output = self.layer_norm(input)
        output, _ = self.mhsa(output, output, output,
                              key_padding_mask = attn_padding_mask, need_weights = False)
        output = self.dropout(output)
        output += input
        return output


def test_mhsa_module():
    batch_size = 13
    n_timesteps = 100
    num_heads = 7
    in_features = 119
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(batch_size, n_timesteps, in_features).to(device)
    model = MHSAModule(in_features, num_heads).to(device)
    output = model(input)
    assert output.shape == input.shape

