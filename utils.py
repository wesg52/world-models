import datetime
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


MODEL_FAMILIES = ['pythia', 'Llama-2']


def get_model_family(model_name):
    for family in MODEL_FAMILIES:
        if family in model_name:
            return family
    raise ValueError(f'Invalid model name: {model_name}')


def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")


def adjust_precision(activation_tensor, output_precision=8, per_channel=True, cos_sim=False):
    '''
    Adjust the precision of the activation subset
    '''
    if output_precision == 64:
        return activation_tensor.to(torch.float64)

    elif output_precision == 32:
        return activation_tensor.to(torch.float32)

    elif output_precision == 16:
        return activation_tensor.to(torch.float16)

    elif output_precision == 8 and not per_channel:
        min_val = activation_tensor.min().item() if not cos_sim else -1
        max_val = activation_tensor.max().item() if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_val - min_val) / (num_quant_levels - 1)
        zero_point = round(-min_val / scale)
        return torch.quantize_per_tensor(activation_tensor, scale, zero_point, torch.quint8)

    elif output_precision == 8 and per_channel:
        min_vals = activation_tensor.min(dim=0)[0] if not cos_sim else -1
        max_vals = activation_tensor.max(dim=0)[0] if not cos_sim else 1
        num_quant_levels = 2**output_precision
        scale = (max_vals - min_vals) / (num_quant_levels - 1)
        zero_point = torch.round(-min_vals / scale)
        return torch.quantize_per_channel(activation_tensor, scale, zero_point, 1,  torch.quint8)

    else:
        raise ValueError(f'Invalid output precision: {output_precision}')


MODEL_N_LAYERS = {
    'pythia-70m': 6,
    'pythia-160m': 12,
    'pythia-410m': 24,
    'pythia-1b': 16,
    'pythia-1.4b': 24,
    'pythia-2.8b': 32,
    'pythia-6.9b': 32,
    'Llama-2-7b-hf': 32,
    'Llama-2-13b-hf': 40,
    'Llama-2-70b-hf': 80,
}


LLAMA_70B_SUPERCLOUD_DEVICE_MAP = {
    'model.embed_tokens': 0,
    'model.layers.0': 0,
    'model.layers.1': 0,
    'model.layers.2': 0,
    'model.layers.3': 0,
    'model.layers.4': 0,
    'model.layers.5': 0,
    'model.layers.6': 0,
    'model.layers.7': 0,
    'model.layers.8': 0,
    'model.layers.9': 0,
    'model.layers.10': 0,
    'model.layers.11': 0,
    'model.layers.12': 0,
    'model.layers.13': 0,
    'model.layers.14': 0,
    'model.layers.15': 0,
    'model.layers.16': 0,
    'model.layers.17': 0,
    'model.layers.18': 0,
    'model.layers.19': 0,
    'model.layers.20': 0,
    'model.layers.21': 0,
    'model.layers.22': 0,
    'model.layers.23': 0,
    'model.layers.24': 0,
    'model.layers.25': 0,
    'model.layers.26': 0,
    'model.layers.27': 0,
    'model.layers.28': 0,
    'model.layers.29': 0,
    'model.layers.30': 0,
    'model.layers.31': 0,
    'model.layers.32': 0,
    'model.layers.33': 0,
    'model.layers.34': 0,
    'model.layers.35': 'cpu',
    'model.layers.36': 'cpu',
    'model.layers.37': 'cpu',
    'model.layers.38': 'cpu',
    'model.layers.39': 'cpu',
    'model.layers.40': 'cpu',
    'model.layers.41': 'cpu',
    'model.layers.42': 'cpu',
    'model.layers.43': 'cpu',
    'model.layers.44': 'cpu',
    'model.layers.45': 'cpu',
    'model.layers.46': 'cpu',
    'model.layers.47': 'cpu',
    'model.layers.48': 'cpu',
    'model.layers.49': 'cpu',
    'model.layers.50': 'cpu',
    'model.layers.51': 'cpu',
    'model.layers.52': 'cpu',
    'model.layers.53': 'cpu',
    'model.layers.54': 'cpu',
    'model.layers.55': 'cpu',
    'model.layers.56': 'cpu',
    'model.layers.57': 'cpu',
    'model.layers.58': 'cpu',
    'model.layers.59': 'cpu',
    'model.layers.60': 'cpu',
    'model.layers.61': 'cpu',
    'model.layers.62': 'cpu',
    'model.layers.63': 'cpu',
    'model.layers.64': 'cpu',
    'model.layers.65': 'cpu',
    'model.layers.66': 'cpu',
    'model.layers.67': 'cpu',
    'model.layers.68': 'cpu',
    'model.layers.69': 'cpu',
    'model.layers.70': 'cpu',
    'model.layers.71': 'cpu',
    'model.layers.72': 'cpu',
    'model.layers.73': 'cpu',
    'model.layers.74': 'cpu',
    'model.layers.75': 'cpu',
    'model.layers.76': 'cpu',
    'model.layers.77': 'cpu',
    'model.layers.78': 'cpu',
    'model.layers.79': 'cpu',
    'model.norm': 'cpu',
    'lm_head': 'cpu'
}


LLAMA_70B_8BIT_SUPERCLOUD_DEVICE_MAP = {
    'model.embed_tokens': 0,
    'model.layers.0': 0,
    'model.layers.1': 0,
    'model.layers.2': 0,
    'model.layers.3': 0,
    'model.layers.4': 0,
    'model.layers.5': 0,
    'model.layers.6': 0,
    'model.layers.7': 0,
    'model.layers.8': 0,
    'model.layers.9': 0,
    'model.layers.10': 0,
    'model.layers.11': 0,
    'model.layers.12': 0,
    'model.layers.13': 0,
    'model.layers.14': 0,
    'model.layers.15': 0,
    'model.layers.16': 0,
    'model.layers.17': 0,
    'model.layers.18': 0,
    'model.layers.19': 0,
    'model.layers.20': 0,
    'model.layers.21': 0,
    'model.layers.22': 0,
    'model.layers.23': 0,
    'model.layers.24': 0,
    'model.layers.25': 0,
    'model.layers.26': 0,
    'model.layers.27': 0,
    'model.layers.28': 0,
    'model.layers.29': 0,
    'model.layers.30': 0,
    'model.layers.31': 0,
    'model.layers.32': 0,
    'model.layers.33': 0,
    'model.layers.34': 0,
    'model.layers.35': 1,
    'model.layers.36': 1,
    'model.layers.37': 1,
    'model.layers.38': 1,
    'model.layers.39': 1,
    'model.layers.40': 1,
    'model.layers.41': 1,
    'model.layers.42': 1,
    'model.layers.43': 1,
    'model.layers.44': 1,
    'model.layers.45': 1,
    'model.layers.46': 1,
    'model.layers.47': 1,
    'model.layers.48': 1,
    'model.layers.49': 1,
    'model.layers.50': 1,
    'model.layers.51': 1,
    'model.layers.52': 1,
    'model.layers.53': 1,
    'model.layers.54': 1,
    'model.layers.55': 1,
    'model.layers.56': 1,
    'model.layers.57': 1,
    'model.layers.58': 1,
    'model.layers.59': 1,
    'model.layers.60': 1,
    'model.layers.61': 1,
    'model.layers.62': 1,
    'model.layers.63': 1,
    'model.layers.64': 1,
    'model.layers.65': 1,
    'model.layers.66': 1,
    'model.layers.67': 1,
    'model.layers.68': 1,
    'model.layers.69': 1,
    'model.layers.70': 1,
    'model.layers.71': 1,
    'model.layers.72': 'cpu',
    'model.layers.73': 'cpu',
    'model.layers.74': 'cpu',
    'model.layers.75': 'cpu',
    'model.layers.76': 'cpu',
    'model.layers.77': 'cpu',
    'model.layers.78': 'cpu',
    'model.layers.79': 'cpu',
    'model.norm': 'cpu',
    'lm_head': 'cpu'
}
