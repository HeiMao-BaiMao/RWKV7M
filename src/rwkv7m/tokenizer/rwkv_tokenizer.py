import ast
from importlib.resources import files
from pathlib import Path


class Trie:
    __slots__ = ("ch", "to", "values", "front")

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            self.values.add(key if val is None else val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = Trie(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        node = self
        ch = key[idx]
        ret = None
        while node.to[ch] is not None:
            node = node.to[ch]
            idx += 1
            if node.values:
                ret = idx, node.values
            if idx == len(key):
                break
            ch = key[idx]
        if ret is None:
            raise ValueError(f"tokenizer could not encode byte at offset {idx}")
        return ret


def default_vocab_path():
    repo_copy = Path(__file__).resolve().parents[3] / "data" / "rwkv_vocab_v20230424.txt"
    if repo_copy.exists():
        return repo_copy

    asset = files("rwkv7m.assets").joinpath("rwkv_vocab_v20230424.txt")
    if asset.is_file():
        return Path(str(asset))
    return repo_copy


class RWKVTokenizer:
    def __init__(self, vocab_file=None):
        self.vocab_file = Path(vocab_file) if vocab_file is not None else default_vocab_path()
        if not self.vocab_file.exists():
            raise FileNotFoundError(
                f"vocab file not found: {self.vocab_file}. "
                "Expected the repository copy at data/rwkv_vocab_v20230424.txt, "
                "or pass vocab_file explicitly."
            )

        self.idx2token = {}
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[: line.index(" ")])
                token = ast.literal_eval(line[line.index(" ") : line.rindex(" ")])
                token = token.encode("utf-8") if isinstance(token, str) else token
                if not isinstance(token, bytes):
                    raise ValueError(f"invalid token at vocab index {idx}: {token!r}")
                self.idx2token[idx] = token

        self.token2idx = {token: int(idx) for idx, token in self.idx2token.items()}
        self.root = Trie()
        for token, idx in self.token2idx.items():
            self.root.add(token, val=(token, idx))

    @property
    def vocab_size(self):
        return max(self.idx2token) + 1 if self.idx2token else 0

    def encode_bytes(self, src: bytes):
        idx = 0
        tokens = []
        while idx < len(src):
            next_idx, values = self.root.find_longest(src, idx)
            _, token_id = next(iter(values))
            tokens.append(token_id)
            idx = next_idx
        return tokens

    def decode_bytes(self, tokens):
        return b"".join(self.idx2token[int(token)] for token in tokens)

    def encode(self, text: str, *, add_eos: bool = False):
        tokens = self.encode_bytes(text.encode("utf-8"))
        if add_eos:
            tokens.append(0)
        return tokens

    def decode(self, tokens, *, errors: str = "replace"):
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)
