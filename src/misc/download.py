import argparse
import gdown

parser = argparse.ArgumentParser()
parser.add_argument('run_name')


file_to_url = {
    'conformer-train-clean-100-beam-search-3-gram.zip': 'https://drive.google.com/file/d/1DikeH61alDP5wu0AfDDkiDmc98xXtjxo/view?usp=sharing',
    'conformer-train-other-beam-search-3-gram-continue.zip': 'https://drive.google.com/file/d/1PnLA3aWL1r_N-dKapqL0N1F1S1XJxLW4/view?usp=drive_link',
    'conformer-train-other-beam-search-3-gram.zip': 'https://drive.google.com/file/d/1HjbCC1rymKlkB0kCUPCe-_77kKDZpIvI/view?usp=drive_link',
    'librispeech-3-gram.zip': 'https://drive.google.com/file/d/1s7OMLJ4Q1st-U3-3k5zU2JeAS75UvJZV/view?usp=drive_link',
}


checkpoint = 'conformer-train-other-beam-search-3-gram-continue.zip' # final model
gdown.download(file_to_url[checkpoint], checkpoint, fuzzy=True)

lm = 'librispeech-3-gram.zip'
gdown.download(file_to_url[lm], lm, fuzzy=True)