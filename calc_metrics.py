#!/usr/bin/env python3

import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.metrics.utils import calc_cer, calc_wer

warnings.filterwarnings("ignore", category=UserWarning)

def get_texts(texts_dir):
    texts = {}
    for path in Path(texts_dir).iterdir():
        if path.is_file():
            with open(path) as f:
                texts[path.stem] = f.read().strip().lower()
    return texts

@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    """
    Script is used to calculate metrics given the predicted texts and model's parameters
    """
    predictions = get_texts(config.predictions_dir)
    transcriptions = get_texts(config.transcriptions_dir)

    assert set(predictions.keys()) == set(transcriptions.keys())

    cers = {key:calc_cer(transcriptions[key], predictions[key]) for key in predictions.keys()}
    wers = {key:calc_wer(transcriptions[key], predictions[key]) for key in predictions.keys()}

    print('cer:', sum(cers.values()) / len(cers))
    print('wer:', sum(wers.values()) / len(wers))

if __name__ == '__main__':
    main()
