import torch
from transformers import LlamaForCausalLM, AutoTokenizer
try:
    from transformer_lens import HookedTransformer
except ImportError:
    pass


def load_model(model_name="pythia-70m", device=None, checkpoint_index=None, use_hf=True, dtype=torch.float32):
    if use_hf:
        if model_name == 'Llama-2-70b-hf':
            hf_model = LlamaForCausalLM.from_pretrained(
                f"meta-llama/{model_name}",
                low_cpu_mem_usage=True,
                device_map='auto',
                load_in_8bit=True,
                torch_dtype=torch.float16
            )
            hf_model.eval()
            print(hf_model.hf_device_map)
        else:
            hf_model = LlamaForCausalLM.from_pretrained(
                f"meta-llama/{model_name}", low_cpu_mem_usage=True, torch_dtype=dtype)
            hf_model.eval()
            hf_model.to(device)
        return hf_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name.startswith("Llama"):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        hf_model = LlamaForCausalLM.from_pretrained(
            f"meta-llama/{model_name}", low_cpu_mem_usage=True)
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, device="cpu",
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            tokenizer=tokenizer
        )
        del hf_model

    else:
        model = HookedTransformer.from_pretrained(
            model_name, device='cpu', checkpoint_index=checkpoint_index)

    model.eval()
    if model.cfg.device != device:
        try:
            model.to(device)
        except RuntimeError:
            print(
                f"WARNING: model is too large to fit on {device}. Falling back to CPU")
            model.to('cpu')

    return model
