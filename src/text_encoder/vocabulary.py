class Vocabulary:
    def __init__(self, tokens = None, silence_token = '|', blank_token = '-', tokenizer = None):
        if tokens is None:
            assert tokenizer is not None
            self.tokens = tokenizer.tokens
        else:
            self.tokens = tokens
            
        self.silence_token = silence_token
        self.blank_token = blank_token
        self.ind2char = dict(enumerate(self.tokens))
        self.char2ind = {v:k for k, v in self.ind2char.items()}
