import re
from string import ascii_lowercase

import torch

# TODO add BPE

class CTCTextEncoder:
    def __init__(self, vocab):
        """
        Args:
            vocab (Vocabulary): alphabet for language.
        """
        self.vocab = vocab
        assert self.vocab.char2ind[self.vocab.blank_token] == 0, \
            "Current implementation relies on index of blank token being 0 (see torch.nn.CTCLoss)"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        text = text.replace(' ', self.silence_token)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    # @lminasian TODO: remove this shit
    # adding it now in case external code calls text_encoder.ind2char etc.
    @property
    def tokens(self):
        return self.vocab.tokens
    
    @property
    def ind2char(self):
        return self.vocab.ind2char
    
    @property
    def char2ind(self):
        return self.vocab.char2ind
    
    @property
    def silence_token(self):
        return self.vocab.silence_token

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^'a-z ]", "", text) # TODO: place here vocab.tokens
        return text
