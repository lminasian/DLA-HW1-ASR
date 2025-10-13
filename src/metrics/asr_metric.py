from typing import List

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer
from dataclasses import dataclass
from src.text_encoder.ctc_decoder import *


@dataclass
class Prediction:
    text: str
    score: float


class ASRMetric(BaseMetric):
    def __init__(self, decoder, eval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder
        self.eval = eval


    def __call__(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor, text: List[str], **kwargs
    ):
        predictions = self.decode_and_eval(log_probs, log_probs_length, text, **kwargs)
        average_score = sum(map(lambda p: p.score, predictions)) / len(predictions)
        return average_score


    def decode_and_eval(self, log_probs: torch.Tensor, log_probs_length: torch.Tensor, text: List[str], **kwargs):
        predictions = []
        for log_prob_vec, length, target_text in zip(log_probs, log_probs_length, text):
            pred_text = self.decoder(log_prob_vec[:length])
            score_for_pred = self.eval(target_text, pred_text)
            predictions.append(Prediction(
                text = pred_text,
                score = score_for_pred,
            ))
        return predictions


class CERMetric(ASRMetric):
    def __init__(self, decoder, *args, **kwargs):
        super().__init__(decoder, eval=calc_cer, *args, **kwargs)


class WERMetric(ASRMetric):
    def __init__(self, decoder, *args, **kwargs):
        super().__init__(decoder, eval=calc_wer, *args, **kwargs)


class ArgmaxCERMetric(CERMetric):
    def __init__(self, decoder, *args, **kwargs):
        assert isinstance(decoder, CTCArgmaxDecoder)
        super().__init__(decoder, *args, **kwargs)


class ArgmaxWERMetric(WERMetric):
    def __init__(self, decoder, *args, **kwargs):
        assert isinstance(decoder, CTCArgmaxDecoder)
        super().__init__(decoder, *args, **kwargs)


class RawCERMetric(CERMetric):
    def __init__(self, decoder, *args, **kwargs):
        assert isinstance(decoder, CTCRawDecoder)
        super().__init__(decoder, *args, **kwargs)


class RawWERMetric(WERMetric):
    def __init__(self, decoder, *args, **kwargs):
        assert isinstance(decoder, CTCRawDecoder)
        super().__init__(decoder, *args, **kwargs)


class BeamSearchCERMetric(CERMetric):
    def __init__(self, decoder, *args, **kwargs):
        assert isinstance(decoder, CTCBeamSearchDecoder)
        super().__init__(decoder)


class BeamSearchWERMetric(WERMetric):
    def __init__(self, decoder, *args, **kwargs):
        assert isinstance(decoder, CTCBeamSearchDecoder)
        super().__init__(decoder)