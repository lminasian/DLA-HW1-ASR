import torch
import torch.nn as nn


# TODO: how does conformer handles padding (and log_probs_lengths)
# if convolutions do use padded elements???

class ConvolutionModule(nn.Module):
    def __init__(self, in_features, hidden_size, depthwise_kernel_size, dropout = 0.0):
        super().__init__()
        assert depthwise_kernel_size % 2 != 0, ""
        self.layer_norm = nn.LayerNorm((in_features,))
        self.pointwise_conv_1 = nn.Conv1d(
            in_channels = in_features,
            out_channels = hidden_size * 2,
            kernel_size = (1,),
        )
        self.glu = nn.GLU(dim = 1)
        self.depthwise_conv = nn.Conv1d(
            in_channels = hidden_size,
            out_channels = hidden_size,
            kernel_size = (depthwise_kernel_size,),
            groups = hidden_size, # !!! should be equal to in_channels for depthwise !!!
            padding = (depthwise_kernel_size - 1) // 2,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(
            in_channels = hidden_size,
            out_channels = in_features,
            kernel_size = (1,),
        )
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, input):
        """
        Args:
            input: Tensor[N, T, C]: input (spectrogram)

        Returns:
        """
        # TODO: replace with nn.Sequential
        output = self.layer_norm(input)
        output = output.transpose(1, 2)

        output = self.pointwise_conv_1(output)
        output = self.glu(output)
        output = self.depthwise_conv(output)
        output = self.batch_norm(output)
        output = self.swish(output)
        output = self.pointwise_conv_2(output)
        output = self.dropout(output)

        output = output.transpose(1, 2)
        output += input
        return output

def test_convolution_module():
    batch_size = 13
    n_timesteps = 100
    in_features = 158
    hidden_size = 14
    depthwise_kernel_size = 25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(batch_size, n_timesteps, in_features).to(device)
    model = ConvolutionModule(in_features, hidden_size, depthwise_kernel_size).to(device)
    output = model(input)
    assert output.shape == input.shape
