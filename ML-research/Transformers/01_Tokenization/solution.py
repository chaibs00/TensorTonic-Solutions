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

        # Add special token to dict :
        self.word_to_id = {
            self.pad_token : 0 ,
            self.unk_token : 1 ,
            self.bos_token : 2 ,
            self.eos_token : 3 ,
        }

        # Collect unique words : 
        words = set()      # using set to save only unique
        for sentence in texts :
            words.update(sentence.lower().split())     # split() by spaces 


        # Build vocab dictionary (word , ID)
        # Assign IDs in alphabetic order :
        for id, word in enumerate( sorted(words), start=4):
            self.word_to_id[word] = id


        # The reverse dictionary (ID, word)
        for key, val in self.word_to_id.items() :
            self.id_to_word[val] = key 

        # The size of vocabulary : 
        self.vocab_size = len(self.word_to_id)

  

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        
        # Split the text into words :
        words = str(text).lower().split()
        
        # Result list 
        ids = []

        # Query the vocabulary 
        for word in words :
            if word in self.word_to_id :
                ids.append(self.word_to_id[word])

            else :
                ids.append(1)    
                
        return ids
        

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """

        sentence = []

        for id in ids : 
            if id in self.id_to_word :
                sentence.append(self.id_to_word[id])

            else : 
                sentence.append(self.unk_token)

        res = " ".join(sentence)

        return res
        