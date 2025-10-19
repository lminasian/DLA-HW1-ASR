#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(description='Run inference with CTC model. Your inputs are logits calculated with logits_only.py and the outputs are model\'s predictions')

parser.add_argument('--dataset-config', type=str, default='custom',
                   help='Dataset configuration for hydra. Should be placed inside src/configs/datasets/')
parser.add_argument('--model', type=str, 
                   default='saved/conformer-train-other-beam-search-3-gram/model_best.pth',
                   help='Path to pretrained model')
parser.add_argument('--model-config', type=str, default='conformer',
                   help='Model configuration for hydra. Should be place inside src/configs/model')
parser.add_argument('--save-path', type=str, default='custom/predictions',
                   help='Path to save predictions into. Would be stored in data/saved/.')
parser.add_argument('--logits-dir', type=str, default='data/saved/custom/logits/test/',
                   help='Directory containing logits')
parser.add_argument('--transcription-dir', type=str, default=None)

args = parser.parse_args()

cmd = [
    'python3', 'inference.py',
    f'datasets={args.dataset_config}',
    'inferencer.metrics_only=True',
    f'datasets.test.logits_dir={args.logits_dir}',
    'dataloader.batch_size=1',
    f'inferencer.from_pretrained={args.model}',
    f'model={args.model_config}',
    f'inferencer.save_path={args.save_path}',
]

if args.transcription_dir is not None:
    cmd += [
        f'datasets.test.transcription_dir={args.transcription_dir}',
    ]

os.system(' '.join(cmd))
