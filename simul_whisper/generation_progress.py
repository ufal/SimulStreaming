class Tokens:
    def __init__(self, tokens):
        self.tokens = tokens

#    def clone(self):
#        return Tokens(self.tokens.clone())

    def __str__(self):
        return str(self.tokens.tolist())

    def __repr__(self):
        return self.__str__()

class BeamTokens(Tokens):
    def __init__(self, tokens, beam_size):
        self.tokens = tokens
        self.beam_size = beam_size

    def clone(self):
        return BeamTokens(self.tokens.clone())

    def __str__(self):
        return f"BeamTokens({self.tokens.tolist()}, beam_size={self.beam_size})"

    def __repr__(self):
        return self.__str__()

    def as_text(self, tokenizer):
        return tokenizer.decode(self.tokens)

class Logits(Tokens):
    def __init__(self, logits):
        super().__init__(logits)

#    def clone(self):
#        return Logits(self.tokens.clone(), self.beam_size)

    def __str__(self):
#        return "abc"
        return f"Logits({self.tokens.shape})"

    def __repr__(self):
        return self.__str__()