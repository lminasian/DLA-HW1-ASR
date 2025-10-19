#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(description='Run inference with CTC model outputting only logits.')

parser.add_argument('--dataset-config', type=str, default='custom',
                   help='Dataset configuration')
parser.add_argument('--model', type=str, 
                   default='saved/conformer-train-other-beam-search-3-gram/model_best.pth',
                   help='Path to pretrained model')
parser.add_argument('--model-config', type=str, default='conformer',
                   help='Model configuration')
parser.add_argument('--save-path', type=str, default='custom/logits',
                   help='Path to save logits')

args = parser.parse_args()

cmd = [
    'python3', 'inference.py',
    f'datasets={args.dataset_config}',
    'inferencer.logits_only=True',
    'dataloader.batch_size=1',
    f'inferencer.from_pretrained={args.model}',
    f'model={args.model_config}',
    f'inferencer.save_path={args.save_path}'
]

os.system(' '.join(cmd))
