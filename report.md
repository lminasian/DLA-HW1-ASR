# 1. FAQ
## How to reproduce my model?
I've chosen to use ConformerEncoder+LSTM model with CTC loss.
I did not use BPE/Subsampling/augmentations during training.
(about **bpe** and **augmentations** please see below).

I have implemented my model in 3 stages.
1. train on train-clean-100 with [conformer-train-clean-100-beam-search-3-gram](src/configs/conformer-train-clean-100-beam-search-3-gram.yaml) for 26 epochs.
1. choose the last checkpoint from train-clean-100 and fit it on train-other-500 for another 60 epochs. [conformer-train-other-beam-search-3-gram.yaml](src/configs/conformer-train-other-beam-search-3-gram.yaml)
1. continue on train-other for another 30 epochs with config [conformer-train-other-beam-search-3-gram-continue.yaml](src/configs/conformer-train-other-beam-search-3-gram-continue.yaml)

I set the `epoch_len=2000` and `batch_size=16` for all my runs. I turn beamsearch off during training because it has zero effect on training with CTC-loss, yet takes a lot of time comparing to argmax/raw decoding.

## Attach training logs to show how fast did you network train
1. [conformer-train-clean-100](https://www.comet.com/lminasian/dla-hw1-asr/2nib5oweln3i5tegl09s8bw8fz96nt2i?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
2. [conformer-train-other 60 epochs](https://www.comet.com/lminasian/dla-hw1-asr/x59fvqprl2bcdh3wvgaxtcy8c7h9w2ps?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)
3. [conformer-train-other 30 more epochs](https://www.comet.com/lminasian/dla-hw1-asr/mnkma3jppqpdcn9738v1kzqx868ulmkh?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)


## How did you train your final model
Choose the pretraining model from above. Then I use beamsearch with `beamsize=50` and for lm I chose 3-gram model from pytorch.

To tune the `lm_weight` run the [tune_lm_weight.sh](tune_lm_weight.sh).
(See logs in [tune_lm_weight.log](tune_lm_weight.log) [tune_lm_weight_2_3_11.log](tune_lm_weight_2_3_11.log))

## What have you tried
I tried
* deepspeech (got to conformer because in paper they told `wer=4` without lm)
* training conformer for another 30 epochs to get below `wer=40`

## What worked and what did not
1. Training conformer for another 30 epochs to get from `wer=45` below `wer=40` on train-other did not work.

## What were the major challenges

The biggest challenge was to go beyond `WER_(Argmax)=40` on test-other (the 3rd stage, unsuccessful).
For some reason loss started to converge around `0.3` and there is not that much
I can do about it with time I have left myself.

I would try to:
1. change spectrogram to powerdb scale (too late)
2. implement BPE model and add subsampling
3. train with augmentations
4. add more data from CommonVoice dataset

All the above mentioned points aren't possible to implement now, because all of them 
take time to train model(and all of them take time to be implemented), but I got to work too late(I actually set doing this homework almost immediately, but was doing things very slowly and relaxed. Все, больше не оправдываюсь).

## Conduct some analysis on your plots and experiments
I don't think I should say anything about my first 2 stages, because they both were rather successful.

About 3rd stage: I think things could go better **if** I added **augmentations/bpe/dropout**.
Or maybe the lstm at decoder stage could be deeper(not 1 layer).

If you see the graph of learning-rate on my 3rd stage, you will notice that learning rate was only increasing. So the lr is not the problem.
About gradient: gradient has persisten magnitude of 1.5 on average.

**Loss on test/validation is on average 2 times more** than loss on train. The same things happen with cer/wer, so I conclude the **overfit happen**. The techniques for preventing overfitting(adding more data/augmentations/dropout) could potentially solve the problem for me.

# 2. BONUSES

## BPE
~~TODO: To be added~~

## LM
I have implemented a beam-search with language model (both from pytorch).


# 3. MANDATORY TASKS:

## Beam-search implementation

See the [beam_search.py](src/text_encoder/beam_search.py)

## Augmentations

* For spectrogram: SpecAug, FrequencyMasking, TimeMasking
* For waveform: PitchShifting, AddNoise, SpeedPerturbation, TimeStretch

**Run which shows that augmentations** work: [click](https://www.comet.com/lminasian/dla-hw1-asr/2whc4xpsusqpiqyaxhkkiahuacypy9xa).
It contains logs and everything.

Config for augmentations: [mel_spectrogram_with_augs.yaml](src/configs/transforms/instance_transforms/mel_spectrogram_with_augs.yaml).
Config for this run: [conformer-train-other-beam-search-3-gram-with-augs.yaml](src/configs/conformer-train-other-beam-search-3-gram-with-augs.yaml).


