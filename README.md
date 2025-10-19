# Setup environment
```bash
git clone https://github.com/lminasian/DLA-HW1-ASR lminasian-dla-hw1-asr
cd lminasian-dla-hw1-asr
git checkout solution

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Inference

## Download model
```bash
python3 src/misc/download.py
!unzip conformer-train-other-beam-search-3-gram-continue.zip
!unzip librispeech-3-gram.zip
```

## Test-clean/Test-other
I destroyed clean:
```bash
python3 inference.py datasets=dev_other_test_other
```

# Validate model

```bash
python3 inference.py \
    inferencer.from_pretrained=conformer-train-other-beam-search-3-gram-continue/model_best.pth \
    datasets=test_other
```

## Custom dataset
Your custom dataset is required to have the following structure:
```
NameOfTheDirectoryWithUtterances
├── audio
│   ├── UtteranceID1.wav # may be flac or mp3
│   ├── UtteranceID2.wav
│   .
│   .
│   .
│   └── UtteranceIDn.wav
└── transcriptions # ground truth, may be omitted, SEE BELOW MY FRIEND
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

Run the inference with
```bash
python3 inference.py datasets=custom_no_transcriptions \
    inferencer.predict_text_only=True \
    inferencer.save_path=data/custom/predictions \
    dataloader.batch_size=1
```

Your predictions will appear in specified directory in the following way:
```
data/custom/predictions:
    UtteranceID1.txt
    UtteranceID2.txt
    etc
```

And then you can calculate metrics with

```bash
python3 calc_metrics.py transcriptions_dir=/path/to/transcriptions predictions_dir=data/custom/predictions
```

# Demo
Please see [this notebook](https://www.kaggle.com/code/futuregrandmaster/demodla/edit)


