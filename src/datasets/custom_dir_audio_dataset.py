from pathlib import Path
import os
from src.datasets.base_dataset import BaseDataset
import torchaudio
import torch

class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, logits_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip().lower()
                else:
                    entry['text'] = '' # TODO: dirty or not?

                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry['audio_len'] = length

                if logits_dir:
                    assert Path(logits_dir).exists(), "Logits directory `%s` does not exist" % logits_dir

                    logits_path = Path(logits_dir) / (path.stem + ".pth")
                    assert logits_path.exists(), "Logits file `%s` does not exist" % logits_path
                    entry['log_probs'] = torch.load(logits_path).to('cpu')
            if len(entry) > 0:
                data.append(entry)

        self.logits_provided = (logits_dir is not None)
        super().__init__(data, *args, **kwargs)
    
    def with_logits(self):
        return self.logits_provided
    
    @classmethod
    def from_root(clazz, path_to_root, *args, **kwargs):
        subdirs = os.listdir(path_to_root)
        if "audio" in subdirs and os.path.isdir(os.path.join(path_to_root, 'audio')):
            audio_dir = os.path.join(path_to_root, "audio")
            transcriptions_dir = None
            if "transcriptions" in subdirs and os.path.isdir(os.path.join(path_to_root, 'transcriptions')):
                transcriptions_dir = os.path.join(path_to_root, "transcriptions")
            return clazz(
                audio_dir = audio_dir,
                transcription_dir = transcriptions_dir,
                *args, **kwargs,
            )
        else:
            raise ValueError("Provided directory does not contain `audio` subdir. Create it and come back later.")

