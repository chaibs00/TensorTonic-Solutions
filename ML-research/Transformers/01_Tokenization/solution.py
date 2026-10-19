class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        pass

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        pass