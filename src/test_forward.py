import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers.utils.import_utils as import_utils
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False

import transformers.cache_utils as cache_utils
if hasattr(cache_utils, "DynamicCache"):
    if not hasattr(cache_utils.DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            if isinstance(past_key_values, cls):
                return past_key_values
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (k, v) in enumerate(past_key_values):
                    cache.update(k, v, layer_idx)
            return cache
        cache_utils.DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(cache_utils.DynamicCache, "get_usable_length"):
        def get_usable_length(self, seq_length=None, layer_idx=0):
            if hasattr(self, "get_seq_length"):
                return self.get_seq_length(layer_idx)
            return 0
        cache_utils.DynamicCache.get_usable_length = get_usable_length

    if not hasattr(cache_utils.DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            legacy_cache = ()
            if hasattr(self, "key_cache"):
                for layer_idx in range(len(self.key_cache)):
                    legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
            elif hasattr(self, "layers"):
                for layer in self.layers:
                    legacy_cache += ((layer.keys, layer.values),)
            return legacy_cache
        cache_utils.DynamicCache.to_legacy_cache = to_legacy_cache

from transformers.modeling_utils import PreTrainedModel
orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys
def safe_get_expanded_tied_weights_keys(self, all_submodels=False):
    if hasattr(self, "_tied_weights_keys") and isinstance(self._tied_weights_keys, list):
        self._tied_weights_keys = {k: k for k in self._tied_weights_keys}
    try:
        return orig_get_expanded(self, all_submodels=all_submodels)
    except Exception:
        return {}
PreTrainedModel.get_expanded_tied_weights_keys = safe_get_expanded_tied_weights_keys

STUDENT_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(STUDENT_ID)

model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")

inputs = tok("Hello world! Extracted JSON output:", return_tensors="pt").to("cuda")
out = model(**inputs)
print("Qwen2.5-1.5B Logits NaN:", torch.isnan(out.logits).any().item())
print("Qwen2.5-1.5B Logits max:", out.logits.abs().max().item())
print("Qwen2.5-1.5B Logits shape:", out.logits.shape)
