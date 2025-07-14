import os
from multiprocessing import Process, Queue
from typing import Iterable, Iterator
from collections import defaultdict
from typing import BinaryIO
import regex as re 

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(text: str, special_tokens: list[str] | None = None) -> list[list[int]]:
    """Pre-tokenize the text into a list of byte strings, splitting by special tokens."""
    # Split the text into parts based on special tokens
    special_tokens = special_tokens or []
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    parts = re.split(pattern, text) 

    pre_tokens = [] 
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        words = re.findall(PAT, part)
        for word in words:
            pre_tokens.append(list(word.encode("utf-8")))
    return pre_tokens

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str], 
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []

    # Initialize the vocabulary
    vocab = {x: bytes([x]) for x in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # Chunk the file into parts that can be pre-tokenized independently
    num_processes = 8 
    chunk_list = []
    with open(input_path, "rb") as f:   
        chunk_boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors = "ignore")
            chunk_list.append(chunk)
    
    # Pre-tokenize each chunk in parallel
    queue = Queue()
    processes = []
    pre_tokens = []
    for chunk in chunk_list:
        p = Process(target = lambda q, c: q.put(pretokenize(c, special_tokens)), args = (queue, chunk))
        processes.append(p)
        p.start()
    
    for _ in range(len(processes)):
        pre_tokens.extend(queue.get())
    
    for p in processes:
        p.join()

    # Create frequency table and index set for pairs of bytes
    frequency_table = defaultdict(int)
    index_set = defaultdict(set)
    for tokenid, pre_token in enumerate(pre_tokens):
        for u, v in zip(pre_token[:-1], pre_token[1:]):
            # u, v are consecutive byte values (integers) in one pre-token
            frequency_table[u, v] += 1
            index_set[u, v].add(tokenid)
    
    # Perform merges until we reach the desired vocabulary size
    num_merges = max(0, vocab_size - len(special_tokens) - 256) 
    merges = [] 
    for i in range(num_merges):
        # find max pair by highest frequency
        max_pair = max(
            frequency_table,
            key = lambda pair: (
                frequency_table[pair],
                vocab[pair[0]].decode("utf-8", errors = "ignore"),
                vocab[pair[1]].decode("utf-8", errors = "ignore")
            )
        )
        u, v = max_pair
        new_tokenid = i + 256 + len(special_tokens)
        vocab[new_tokenid] = vocab[u] + vocab[v]
        merges.append((vocab[u], vocab[v]))

        for pos in index_set[max_pair]:
            pre_pretoken = pre_tokens[pos]
            new_pretoken = []
            j = 0
            prelen = len(pre_pretoken)
            lst = -1 
            while j < prelen:
                if j + 1 < prelen and pre_pretoken[j] == u and pre_pretoken[j + 1] == v:
                    new_pretoken.append(new_tokenid)
                    frequency_table[max_pair] -= 1
                    if j > 0: 
                        frequency_table[lst, u] -= 1
                        frequency_table[lst, new_tokenid] += 1
                        index_set[lst, new_tokenid].add(pos)
                    if j + 2 < prelen:
                        nxt = pre_pretoken[j + 2]
                        frequency_table[v, nxt] -= 1
                        frequency_table[new_tokenid, nxt] += 1
                        index_set[new_tokenid, nxt].add(pos)
                    j += 2
                    lst = new_tokenid
                else: 
                    new_pretoken.append(pre_pretoken[j])
                    lst = pre_pretoken[j]
                    j += 1
            pre_tokens[pos] = new_pretoken

        assert frequency_table[max_pair] == 0, "Frequency of max pair must be zero after merge"
    return vocab, merges
    
def main(): 
    file_path = "../data/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>", "."]
    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)

if __name__ == "__main__": 
    main()


