import argparse
import gdown

parser = argparse.ArgumentParser()
parser.add_argument('run_name')


file_to_url = {
    'conformer-train-clean-100-beam-search-3-gram': 'https://drive.google.com/file/d/1DikeH61alDP5wu0AfDDkiDmc98xXtjxo/view?usp=sharing',
    'conformer-train-other-beam-search-3-gram-continue': 'https://drive.google.com/file/d/1HjbCC1rymKlkB0kCUPCe-_77kKDZpIvI/view?usp=drive_link',
    'conformer-train-other-beam-search-3-gram': 'https://drive.google.com/file/d/1HjbCC1rymKlkB0kCUPCe-_77kKDZpIvI/view?usp=drive_link',
    'librispeech-3-gram': 'https://drive.google.com/file/d/1BrvswKjl_3WncPI8ClhGkXf0IYorLlLg/view?usp=drive_link',
}


checkpoint = 'conformer-train-clean-100-beam-search-3-gram-continue' # final model
gdown.download(file_to_url[checkpoint], checkpoint)

lm = 'librispeech-3-gram'
gdown.download(file_to_url[lm], lm)


