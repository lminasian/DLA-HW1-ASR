import torch
import torch.nn as nn

class FFNModule(nn.Module):
    def __init__(self, in_features, hidden_size, dropout = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout1 = nn.Dropout(p = dropout)
        self.dropout2 = nn.Dropout(p = dropout)
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, in_features)
        self.swish = nn.SiLU()
    
    def forward(self, input):
        output = self.layer_norm(input)

        output = self.fc1(output)
        output = self.swish(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.dropout2(output)

        output /= 2 # half-step residual
        output += input
        return output
    

def test_ffn_module():
    batch_size = 13
    n_timesteps = 100
    in_features = 158
    hidden_size = 179
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(batch_size, n_timesteps, in_features).to(device)
    model = FFNModule(in_features, hidden_size).to(device)
    output = model(input)
    assert output.shape == input.shape

