import torch
from torchaudio.models.decoder import ctc_decoder
from typing import List, Union, Optional
from src.text_encoder.vocabulary import Vocabulary

from src.text_encoder.beam_search import aligned_beam_search


class CTCDecoder:
    def __init__(self, vocab: Vocabulary, name = None):
        if name is None:
            name = type(self).__name__
        self.vocab = vocab
        self.name = name

    def __call__(self, log_probs, lengths = None):
        """
        Arguments:
        log_probs: tensor of shape [T, C] or [N, T, C] of log-probabilities of output tokens for each timestamp
        lengths:
            tensor of shape (N,) of actual lengths of sequences before padding IF log_probs is batched
            None if log_probs is not batched
        """
        if len(log_probs.shape) == 3:
            return self.decode_on_batch(log_probs, lengths)
        return self.decode(log_probs)
    
    def decode_on_batch(self, log_probs, lengths):
        if len(log_probs.shape) == 2:
            log_probs = log_probs.unsqueeze(0)
        predictions = []
        for log_probs_sample, length in zip(log_probs, lengths):
            decoded_text = self.decode(log_probs_sample[:length])
            predictions.append(decoded_text)
        return predictions

    def decode(self, log_probs):
        raise NotImplementedError('decode() is not implemented in base class. Call derived ones instead')


class CTCRawDecoder(CTCDecoder):
    def decode(self, log_probs):
        argmax_indices = log_probs.argmax(dim = -1).detach().cpu().numpy()
        pred_text = ''.join(self.vocab.ind2char[i] for i in argmax_indices)
        pred_text = pred_text.replace(self.vocab.silence_token, ' ')
        pred_text = pred_text.replace(self.vocab.blank_token, '')
        return pred_text


class CTCArgmaxDecoder(CTCDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, log_probs):
        argmax_indices = log_probs.argmax(dim = -1)
        unique_consecutive = []
        cur = None
        for i in argmax_indices:
            if cur is None or cur != i:
                unique_consecutive.append(i.item())
                cur = i
        predicted_text = \
            ''.join(self.vocab.ind2char[i] for i in unique_consecutive if self.vocab.ind2char[i] != self.vocab.blank_token)
        predicted_text = predicted_text.replace(self.vocab.silence_token, ' ') # for metrics; the reverse of what text_encoder does
        return predicted_text


class CTCBeamSearchDecoder(CTCDecoder):
    def __init__(self, vocab, lexicon = None, lm = None, lm_weight = 0, nbest = 1, *args, **kwargs):
        assert nbest == 1
        super().__init__(vocab, *args, **kwargs)
        self.torch_beam_search_decoder = ctc_decoder(
            lexicon = lexicon,
            tokens = vocab.tokens,
            lm = lm,
            lm_weight = lm_weight,
            nbest = nbest,
            sil_token = self.vocab.silence_token,
            blank_token = self.vocab.blank_token,
        )
        self.nbest = nbest

    def decode(self, log_probs) -> List:
        assert self.nbest == 1, "Too big appetite my friend"
        lengths = torch.tensor([log_probs.shape[0]])
        hypotheses = self.torch_beam_search_decoder(
            log_probs.unsqueeze(0).detach().cpu().contiguous(),
            lengths.detach().cpu())[0] # batch_size=1
        predicted_text = ' '.join(hypotheses[0].words)
        
        return predicted_text
    

class CTCMyOwnBeamSearchDecoder(CTCDecoder):
    def __init__(self, vocab, nbest = 1, beam_size = 50, *args, **kwargs):
        assert nbest == 1
        super().__init__(vocab, *args, **kwargs)
        self.nbest = nbest
        self.beam_size = beam_size
        
    def decode(self, log_probs) -> List:
        assert self.nbest == 1, "Too big appetite my friend"

        hypothesis = aligned_beam_search(
            log_probs,
            ind2char = self.vocab.ind2char,
            beam_size = self.beam_size,
            empty_token_id = self.vocab.char2ind[self.vocab.blank_token],
            num_candidates = 1
        )[0]
        predicted_text = ' '.join(hypothesis.text.split(self.vocab.silence_token))
        return predicted_text
        


# lminasian TODO: remove code-duplication using pytest features.
# lminasian TODO: set adequate thresholds and name them.

def test_raw_decoder():
    tokens = (
        ['-', '|'] + list(chr(ord('a') + i) for i in range(0, 26))
    )
    vocab = Vocabulary(tokens, silence_token = '|', blank_token = '-')
    raw_decoder = CTCRawDecoder(vocab)
    text = 'ronaldo|is|the|most|famous|guy|in|portugal'
    torch.manual_seed(42)
    log_probs = torch.randn(len(text), len(tokens))
    for i in range(len(text)):
        log_probs[i, vocab.char2ind[text[i]]] = torch.max(log_probs[i]) + 1
    pred_text = raw_decoder.decode(log_probs)
    assert pred_text == text


def test_argmax_decoder():
    tokens = (
        ['-', '|'] + list(chr(ord('a') + i) for i in range(0, 26))
    )
    vocab = Vocabulary(tokens, silence_token = '|', blank_token = '-')
    argmax_decoder = CTCArgmaxDecoder(vocab)
    text = 'ronaldo|is|the|most|famous|guy|in|portugal'
    torch.manual_seed(42)
    log_probs = torch.randn(len(text), len(tokens))
    for i in range(len(text)):
        log_probs[i, vocab.char2ind[text[i]]] = torch.max(log_probs[i]) + 1
    pred_text = argmax_decoder.decode(log_probs)
    assert pred_text == text


def test_beamsearch_decoder(): # may take some time to download LM (only once)
    lexicon = "/home/enakin/.cache/torch/hub/torchaudio/decoder-assets/librispeech-3-gram/lexicon.txt"
    lm = "/home/enakin/.cache/torch/hub/torchaudio/decoder-assets/librispeech-3-gram/lm.bin"
    lm_weight = 3

    tokens = (
        ['-', '|', "'"] + list(chr(ord('a') + i) for i in range(0, 26))
    )
    vocab = Vocabulary(tokens, silence_token = '|', blank_token = '-')

    beam_search_decoder = CTCBeamSearchDecoder(vocab, lexicon, lm, lm_weight = lm_weight)
    text = 'ronaldo|is|the|most|famous|guy|in|portugal'
    torch.manual_seed(42)
    log_probs = torch.randn(len(text), len(tokens))
    for i in range(len(text)):
        log_probs[i, vocab.char2ind[text[i]]] = torch.max(log_probs[i]) + 200
    pred_text = beam_search_decoder.decode(log_probs)
    assert pred_text == text


def test_my_own_beamsearch_decoder(): # may take some time to download LM (only once)
    tokens = (
        ['-', '|', "'"] + list(chr(ord('a') + i) for i in range(0, 26))
    )
    vocab = Vocabulary(tokens, silence_token = '|', blank_token = '-')

    beam_search_decoder = CTCMyOwnBeamSearchDecoder(vocab)
    text = 'ronaldo|is|the|most|famous|guy|in|portugal'
    torch.manual_seed(42)
    log_probs = torch.randn(len(text), len(tokens))
    for i in range(len(text)):
        log_probs[i, vocab.char2ind[text[i]]] = torch.max(log_probs[i]) + 200
    pred_text = beam_search_decoder.decode(log_probs)
    assert pred_text == text