from bpeasy.tokenizer import BPEasyTokenizer

with open('enwik9', 'r') as f:
    text = f.read()

regex_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

tokenizer = BPEasyTokenizer.train(
    iter([text]), # iterator over str
    vocab_size=50256,
    max_token_length=1024,
    regex_pattern=regex_pattern,
    special_tokens=[],
    fill_to_nearest_multiple_of_eight=False,
    name="bpeasy",
)

tokenizer.save("tokenizer.json")
