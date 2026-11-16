"""merge.empty_model gives in-memory models the same rotary frequencies as a fresh model."""
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import circuit as F
from qfs import merge as M


def check():
    cfg = F.config()
    fresh = Qwen3_5ForCausalLM(cfg).model.rotary_emb
    empty = M.empty_model(cfg).model.rotary_emb
    assert empty.inv_freq.dtype == fresh.inv_freq.dtype == torch.float32
    assert torch.equal(empty.inv_freq, fresh.inv_freq), empty.inv_freq[:4]
    assert torch.equal(empty.original_inv_freq, fresh.original_inv_freq)
    assert empty.attention_scaling == fresh.attention_scaling
    print({"rotary": "PASS", "inv_freq[:2]": empty.inv_freq[:2].tolist()})


if __name__ == "__main__":
    check()
