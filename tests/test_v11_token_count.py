from sql_agent.v11_data import _token_count
from sql_agent.v11_training import MAX_LENGTH


class _Encoding(dict):
    """Stands in for transformers.BatchEncoding, which is a dict subclass."""


class _Tokenizer:
    def __init__(self, output):
        self.output = output

    def apply_chat_template(self, value, **kwargs):
        return self.output


def test_token_count_reads_input_ids_not_encoding_keys():
    encoding = _Encoding(input_ids=list(range(2500)), attention_mask=[1] * 2500)
    assert _token_count(_Tokenizer(encoding), [], "SELECT 1") == 2500


def test_token_count_accepts_batched_and_plain_ids():
    assert _token_count(_Tokenizer(_Encoding(input_ids=[list(range(9))])), [], "SELECT 1") == 9
    assert _token_count(_Tokenizer(list(range(7))), [], "SELECT 1") == 7


def test_training_length_covers_measured_v11_maximum():
    assert MAX_LENGTH >= 2900
