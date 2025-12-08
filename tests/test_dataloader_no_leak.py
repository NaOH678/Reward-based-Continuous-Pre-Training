import torch
import pytest

from flame.data import DataCollatorForLanguageModeling


class FakeTokenizer:
    def __init__(self, vocab_size=100, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def __call__(self, texts, return_attention_mask=False):
        # texts are already list of token ids
        return {"input_ids": texts}

    def pad(self, batch, return_tensors=None, return_attention_mask=False):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        for x in batch:
            ids = x["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collator_labels_mask_no_leak():
    tok = FakeTokenizer()
    collator = DataCollatorForLanguageModeling(tokenizer=tok)
    batch = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    out = collator(batch)

    # labels should equal input_ids except padding positions masked to -100
    assert torch.equal(out["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.equal(out["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(out["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    # Cross-entropy on uniform logits should be close to log(vocab_size) over valid tokens
    logits = torch.zeros_like(out["input_ids"], dtype=torch.float).unsqueeze(-1).expand(-1, -1, tok.vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)
    labels_flat = out["labels"].view(-1)
    mask = labels_flat != -100
    chosen = log_probs.view(-1, tok.vocab_size)[mask, labels_flat[mask]]
    ce = -chosen.mean()  # should be ~ log(V)
    assert ce.item() == pytest.approx(torch.log(torch.tensor(tok.vocab_size)).item(), rel=0.05)
