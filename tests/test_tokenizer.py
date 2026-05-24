from rwkv7m import RWKVTokenizer
from rwkv7m.tokenizer import default_vocab_path


def test_default_vocab_exists():
    assert default_vocab_path().exists()


def test_rwkv_tokenizer_roundtrip_ascii_and_unicode():
    tokenizer = RWKVTokenizer()
    text = "Hello, RWKV7M.\nこんにちは。"
    tokens = tokenizer.encode(text)
    assert tokens
    assert tokenizer.decode(tokens) == text


def test_rwkv_tokenizer_adds_eos():
    tokenizer = RWKVTokenizer()
    tokens = tokenizer.encode("abc", add_eos=True)
    assert tokens[-1] == 0
    assert tokenizer.decode(tokens[:-1]) == "abc"
