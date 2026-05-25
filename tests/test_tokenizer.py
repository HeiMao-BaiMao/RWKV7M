from rwkv7m import RWKVTokenizer
from rwkv7m.tokenizer import default_vocab_path, tokenizer_metadata


def test_default_vocab_exists():
    assert default_vocab_path().exists()


def test_tokenizer_metadata_identifies_vocab():
    metadata = tokenizer_metadata()
    assert metadata["tokenizer_format"] == "rwkv_vocab"
    assert metadata["tokenizer_vocab_name"] == "rwkv_vocab_v20230424.txt"
    assert len(metadata["tokenizer_vocab_sha256"]) == 64


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
