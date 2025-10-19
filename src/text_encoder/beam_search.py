import torch
from typing import Dict, Tuple
from dataclasses import dataclass
import itertools
from collections import defaultdict
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

@dataclass(frozen = True)
class BeamState:
    prefix: str = ""
    ends_with_empty: bool = False


@dataclass(frozen = True)
class Hypothesis:
    text: str = ""
    score: float = 0


def expand_beam(state: BeamState, state_proba: float,
        ind2char: Dict[str, int], char_proba: torch.Tensor, empty_token_id: int) -> Dict[BeamState, float]:
    """
    private use
    """
    
    def append(state: BeamState, idx: int):
        if idx == empty_token_id:
            if state.ends_with_empty:
                return BeamState(state.prefix, True)
            else:
                return BeamState(state.prefix + ind2char[idx], True)
        else:
            if state.ends_with_empty:
                return BeamState(state.prefix + ind2char[idx], False)
            elif state.prefix == "" or state.prefix[-1] != ind2char[idx]:
                return BeamState(state.prefix + ind2char[idx], False)
            else:
                return BeamState(state.prefix, False)

    new_states = {}
    for i in ind2char:
        new_state = append(state, i)
        new_states[new_state] = state_proba + char_proba[i]
    return new_states


def aligned_beam_search(log_probs, ind2char, beam_size, empty_token_id, num_candidates = 1):
    """
    Arguments:
        log_probs: tensor[seq_len, num_tokens]: log-probabilities of predictions at each timestamp.
        ind2char: vocabulary
        beam_size: int
        empty_token_id: int
        num_candidates: int

    Returns:
        batched_candidates: 2d-list of Tuple[str, proba] of size [batch_size, num_candidates]:
            contains predicted candidates for each sample of the batch
    """
    probs = log_probs

    seq_len, _ = log_probs.shape
    beam_states: defaultdict[BeamState, float] = defaultdict(float)
    beam_states[BeamState('', False)] = 0
    for t in range(seq_len):
        new_beam_states = defaultdict(float)
        for state, state_proba in beam_states.items():
            new_states = expand_beam(state, state_proba, ind2char, probs[t], empty_token_id)
            for new_state, new_state_proba in new_states.items():
                new_beam_states[new_state] = torch.logsumexp(
                    torch.tensor([new_beam_states[new_state], new_state_proba]), dim=0)
        beam_states = dict(sorted(list(new_beam_states.items()), key = lambda s: -s[1])[:beam_size])

    hypotheses = list(map(
        lambda ss: Hypothesis(ss[0].prefix.replace(ind2char[empty_token_id], ''), ss[1]),
        beam_states.items()
    ))
    hypotheses_sorted: list[Hypothesis] = sorted(hypotheses, key = lambda h: -h.score)
    return hypotheses_sorted


def argmax_search(log_probs, ind2char, blank_token = '-', silence_token = ' '):
    argmax_indices = log_probs.argmax(dim = -1)
    score = torch.sum(log_probs.max(dim = -1).values)

    unique_consecutive = []
    cur = None
    for i in argmax_indices:
        if cur is None or cur != i:
            unique_consecutive.append(i.item())
            cur = i
    predicted_text = \
        ''.join(ind2char[i] for i in unique_consecutive if ind2char[i] != blank_token)
    
    return [Hypothesis(predicted_text, score)]


def stresstest_beam_search_with_argmax():
    ind2char = {
        0: '-',
        1: '|',
        2: 'a',
        3: 'b',
        4: 'c',
        5: 'd',
        6: 'e',
        7: 'f',
        8: 'g',
        9: 'h',
    }

    blank_token = '-'
    silence_token = '|'

    torch.manual_seed(42)

    n_tests = 1000
    n_timesteps = 10
    vocab_size = len(ind2char)
    for i_test in range(n_tests):
        print(f'i_test: {i_test}')
        log_probs = torch.rand(n_timesteps, vocab_size)
        hypotheses = aligned_beam_search(log_probs, ind2char, 1, 0, num_candidates = 1)
        best_hypothesis = hypotheses[0]
        best_hypothesis.text, best_hypothesis.score

        argmax_hypothesis = argmax_search(log_probs, ind2char, blank_token, silence_token)[0]
        
        assert best_hypothesis.text == argmax_hypothesis.text


def test_my_beam_search(): # may take some time to download LM (only once)
    tokens = (
        ['-', '|', "'"] + list(chr(ord('a') + i) for i in range(0, 26))
    )
    empty_token_id = 0
    ind2char = dict(enumerate(tokens))
    char2ind = {v:k for k, v in ind2char.items()}

    text = 'ronaldo|is|the|most|famous|guy|in|portugal'
    torch.manual_seed(42)
    log_probs = torch.randn(len(text), len(tokens))
    for i in range(len(text)):
        log_probs[i, char2ind[text[i]]] = torch.max(log_probs[i]) + 200

    beam_size = 50
    beam_hypothesis = aligned_beam_search(log_probs, ind2char, beam_size, empty_token_id, num_candidates=1)[0]
    assert beam_hypothesis.text == text


def test_my_beam_search_1():
    from src.metrics.utils import calc_cer, calc_wer
    tokens = (
        ['-', '|', "'"] + list(chr(ord('a') + i) for i in range(0, 26))
    )
    transcription = 'HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE'.lower()
    empty_token_id = 0
    ind2char = dict(enumerate(tokens))
    char2ind = {v:k for k, v in ind2char.items()}
    
    beam_size = 10

    logit_path = 'data/saved/custom/logits/test/1089-134686-0000.pth'
    log_probs = torch.load(logit_path).to('cpu')
    beam_hypothesis = aligned_beam_search(log_probs, ind2char, beam_size, empty_token_id, num_candidates=1)[0]
    print(beam_hypothesis.text)
    assert calc_cer(transcription, beam_hypothesis.text) < 50


def test_better_than_argmax():
    log_probs = torch.log(torch.tensor([
        [0.5, 0.4, 0.3], # a
        [0.4, 0.5, 0.2], # b
        [0.1, 0.1, 0.5], # -
    ])).T

    ind2char = {
        0: 'a',
        1: 'b',
        2: '-'
    }
    empty_token_id = 2

    # argmax is `ab-`
    # optimal is `bbb`

    beam_size = 50

    argmax_hypothesis = argmax_search(log_probs, ind2char, blank_token='-', silence_token=None)[0]
    beamsearch_hypothesis = aligned_beam_search(log_probs, ind2char, beam_size, empty_token_id)[0]
    assert argmax_hypothesis.text == 'ab'
    assert beamsearch_hypothesis.text == 'b'
