from .train_bpe import pretokenize  
from typing import Iterable, Iterator

class BPE_Tokenizer: 
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None): 
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {token: id for id, token in vocab.items()}
        self.id_to_token = vocab 
        self.byte_special_tokens = [special_token.encode("utf-8") for special_token in self.special_tokens]

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "BPE_Tokenizer":
        """Load a BPE tokenizer from vocab and merges files."""
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        with open(merges_path, 'r') as f:
            merges = [tuple(line.strip().split()) for line in f]
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]: 
        byte_pre_tokens = pretokenize(text, self.special_tokens, False)
        pre_tokens = []
        # Note special tokens are not split into multiple tokens
        for byte_pre_token in byte_pre_tokens:
            if byte_pre_token in self.byte_special_tokens:
                pre_tokens.append([self.token_to_id[byte_pre_token]])
            else: 
                pre_token = []
                for byte_token in byte_pre_token: 
                    pre_token.append(self.token_to_id[bytes([byte_token])])
                pre_tokens.append(pre_token)
        
        # Merge consecutive tokens based on BPE merges
        for i, pre_token in enumerate(pre_tokens): 
            if len(pre_token) == 0: 
                continue  
            for byte1, byte2 in self.merges: 
                new_pre_token = []
                j = 0
                new_token_id = self.token_to_id[byte1 + byte2]
                while j < len(pre_token): 
                    if j < len(pre_token) - 1 and self.id_to_token[pre_token[j]] == byte1 and self.id_to_token[pre_token[j + 1]] == byte2:
                        new_pre_token.append(new_token_id)
                        j += 2
                    else:
                        new_pre_token.append(pre_token[j])
                        j += 1
                pre_token = new_pre_token
            pre_tokens[i] = pre_token
        tokens = [token for pre_token in pre_tokens for token in pre_token]
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.

        usage: 
        with open("large.txt", "r", encoding="utf-8") as f:
            for token in tokenizer.encode_iterable(f):
                # do something with token
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        byte_tokens = [self.id_to_token[id] for id in ids]
        return (b"".join(byte_tokens)).decode("utf-8", errors="ignore")    