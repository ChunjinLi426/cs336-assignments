
class BPE_Tokenizer: 
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None): 
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "BPE_Tokenizer":
        """Load a BPE tokenizer from vocab and merges files."""
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        with open(merges_path, 'r') as f:
            merges = [tuple(line.strip().split()) for line in f]
        return cls(vocab, merges, special_tokens)
    

def main(): 
    # do something... 

if __name__ == "__main__":
    main()
