from torch import nn
from torchaudio.models import Conformer
import torch

# uses pytorch conformer

class ConformerWrapper(nn.Module):
    """
    Wrapper of torchaudio's deepspeech
    """

    def __init__(self, n_tokens, input_dim, num_heads, ffn_dim, num_layers, kernel_size,
                 lstm_num_layers, lstm_hidden_size, dropout = 0.0):
        """
        Args:
            n_feats (int): number of input features.
        """
        super().__init__()
        self.conformer_encoder = Conformer(
            input_dim = input_dim,
            num_heads = num_heads,
            ffn_dim = ffn_dim,
            num_layers = num_layers,
            depthwise_conv_kernel_size = kernel_size,
            dropout = dropout,
        )
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            dropout = dropout,
            proj_size = n_tokens,
        )

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram_length = spectrogram_length.to(spectrogram.device)
        output, _ = self.conformer_encoder(spectrogram.transpose(1, 2), spectrogram_length)
        logits, _ = self.lstm(output)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
