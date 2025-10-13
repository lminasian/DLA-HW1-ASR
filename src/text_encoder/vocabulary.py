class Vocabulary:
    def __init__(self, tokens, silence_token = '|', blank_token = '-'):
        self.tokens = tokens
        self.silence_token = silence_token
        self.blank_token = blank_token
        self.ind2char = dict(enumerate(self.tokens))
        self.char2ind = {v:k for k, v in self.ind2char.items()}
