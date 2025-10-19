#!/bin/bash

set -e

MIN_LM_WEIGHT=$1
MAX_LM_WEIGHT=$2
NUM_STEPS=$3

LOG_FILE=lm_weight_tuning.log

grid_values=`python3 -c "import numpy as np; print(' '.join(map(str, np.linspace($MIN_LM_WEIGHT, $MAX_LM_WEIGHT, $NUM_STEPS))))"`

echo -e "Iterating lm_weight over grid:\n${grid_values[@]}"

# yes, bin-search would work :)

set -x
for lm_weight_value in ${grid_values[@]}; do
    python3 inference.py datasets=dev_other \
        inferencer.from_pretrained=saved/conformer-train-other-beam-search-3-gram-continue/model_best.pth \
        decoders.beam_search_decoder.lm_weight=$lm_weight_value
done
set +x

