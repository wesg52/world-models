import os
import torch
import argparse
from utils import *
from load import load_model
from make_prompt_datasets import ENTITY_PROMPTS
from feature_datasets import common

import os
from tqdm import tqdm
import torch
import einops
import numpy as np
from torch.utils.data import DataLoader


def load_activation_probing_dataset_args(args, prompt_name, layer_ix):
    activation_save_path = os.path.join(
        os.getenv('ACTIVATION_DATASET_DIR', 'activation_datasets'),
        args.model,
        args.entity_type
    )
    save_name = f'{args.entity_type}.{args.activation_aggregation}.{prompt_name}.{layer_ix}.pt'
    save_path = os.path.join(activation_save_path, save_name)
    activations = torch.load(save_path)
    return activations


def load_activation_probing_dataset(model, entity_type, prompt_name, layer_ix, activation_aggregation='last'):
    activation_save_path = os.path.join(
        'activation_datasets', model, entity_type)
    save_name = f'{entity_type}.{activation_aggregation}.{prompt_name}.{layer_ix}.pt'
    save_path = os.path.join(activation_save_path, save_name)
    activations = torch.load(save_path)
    return activations


def process_activation_batch(args, batch_activations, step, batch_mask=None):
    cur_batch_size = batch_activations.shape[0]

    if args.activation_aggregation is None:
        # only save the activations for the required indices
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')  # batch, context, dim
        processed_activations = batch_activations[batch_mask]

    if args.activation_aggregation == 'last':
        last_ix = batch_activations.shape[1] - 1
        batch_mask = batch_mask.to(int)
        last_entity_token = last_ix - \
            torch.argmax(batch_mask.flip(dims=[1]), dim=1)
        d_act = batch_activations.shape[2]
        expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
        processed_activations = batch_activations[
            torch.arange(cur_batch_size).unsqueeze(-1),
            expanded_mask,
            torch.arange(d_act)
        ]
        assert processed_activations.shape == (cur_batch_size, d_act)

    elif args.activation_aggregation == 'mean':
        # average over the context dimension for valid tokens only
        masked_activations = batch_activations * batch_mask
        batch_valid_ixs = batch_mask.sum(dim=1)
        processed_activations = masked_activations.sum(
            dim=1) / batch_valid_ixs[:, None]

    elif args.activation_aggregation == 'max':
        # max over the context dimension for valid tokens only (set invalid tokens to -1)
        batch_mask = batch_mask[:, :, None].to(int)
        # set masked tokens to -1
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = masked_activations.max(dim=1)[0]

    return processed_activations


def save_activation_hook(tensor, hook):
    hook.ctx['activation'] = tensor.detach().cpu().to(torch.float16)


@torch.no_grad()
def get_layer_activations_tl(
    args, model, tokenized_dataset, layers='all', device=None,
):
    if layers == 'all':
        layers = list(range(model.cfg.n_layers))
    if device is None:
        device = model.cfg.device

    hooks = [
        (f'blocks.{layer_ix}.hook_resid_post', save_activation_hook)
        for layer_ix in layers
    ]

    entity_mask = torch.tensor(tokenized_dataset['entity_mask'])

    n_seq, ctx_len = tokenized_dataset['input_ids'].shape
    activation_rows = entity_mask.sum().item() \
        if args.activation_aggregation is None \
        else n_seq

    layer_activations = {
        l: torch.zeros(activation_rows, model.cfg.d_model, dtype=torch.float16)
        for l in layers
    }
    layer_offsets = {l: 0 for l in layers}
    bs = args.batch_size
    dataloader = DataLoader(
        tokenized_dataset['input_ids'], batch_size=bs, shuffle=False)

    for step, batch in enumerate(tqdm(dataloader, disable=False)):
        # clip batch to remove excess padding
        batch_entity_mask = entity_mask[step*bs:(step+1)*bs]
        last_valid_ix = torch.argmax(
            (batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
        batch = batch[:, :last_valid_ix].to(device)
        batch_entity_mask = batch_entity_mask[:, :last_valid_ix]

        model.run_with_hooks(
            batch,
            fwd_hooks=hooks,
            stop_at_layer=max(layers) + 1,
        )

        for lix, (hook_pt, _) in enumerate(hooks):
            batch_activations = model.hook_dict[hook_pt].ctx['activation']

            processed_activations = process_activation_batch(
                args, batch_activations, step, batch_entity_mask)

            offset = layer_offsets[layers[lix]]
            save_rows = processed_activations.shape[0]

            layer_activations[layers[lix]][
                offset:offset+save_rows] = processed_activations

            layer_offsets[layers[lix]] += save_rows

        model.reset_hooks()

    return layer_activations


@torch.no_grad()
def get_layer_activations_hf(
    args, model, tokenized_dataset, layers='all', device=None,
):
    if layers == 'all':
        layers = list(range(model.config.num_hidden_layers))
    if device is None:
        device = model.device

    entity_mask = torch.tensor(tokenized_dataset['entity_mask'])

    n_seq, ctx_len = tokenized_dataset['input_ids'].shape
    activation_rows = entity_mask.sum().item() \
        if args.activation_aggregation is None \
        else n_seq

    layer_activations = {
        l: torch.zeros(activation_rows, model.config.hidden_size,
                       dtype=torch.float16)
        for l in layers
    }
    assert args.activation_aggregation == 'last'  # code assumes this
    offset = 0
    bs = args.batch_size
    dataloader = DataLoader(
        tokenized_dataset['input_ids'], batch_size=bs, shuffle=False)

    for step, batch in enumerate(tqdm(dataloader, disable=False)):
        # clip batch to remove excess padding
        batch_entity_mask = entity_mask[step*bs:(step+1)*bs]
        last_valid_ix = torch.argmax(
            (batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
        batch = batch[:, :last_valid_ix].to(device)
        batch_entity_mask = batch_entity_mask[:, :last_valid_ix]

        out = model(batch, output_hidden_states=True,
                    output_attentions=False, return_dict=True, use_cache=False)

        # do not save post embedding layer activations
        for lix, activation in enumerate(out.hidden_states[1:]):
            if lix not in layer_activations:
                continue
            activation = activation.cpu().to(torch.float16)
            processed_activations = process_activation_batch(
                args, activation, step, batch_entity_mask)

            save_rows = processed_activations.shape[0]
            layer_activations[lix][offset:offset +
                                   save_rows] = processed_activations

        offset += batch.shape[0]

    return layer_activations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--entity_type',
        help='Name of entity_type (should be dir under data/entity_datasets/)')
    parser.add_argument(
        '--activation_aggregation', default='last',
        help='Average activations across all tokens in a sequence')
    # base experiment params
    parser.add_argument(
        '--device', default="cuda" if torch.cuda.is_available() else "cpu",
        help='device to use for computation')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size to use for model.forward')
    parser.add_argument(
        '--save_precision', type=int, default=8, choices=[8, 16, 32],
        help='Number of bits to use for saving activations')
    parser.add_argument(
        '--n_threads', type=int,
        default=int(os.getenv('SLURM_CPUS_PER_TASK', 8)),
        help='number of threads to use for pytorch cpu parallelization')
    parser.add_argument(
        '--layers', nargs='+', type=int, default=None)
    parser.add_argument(
        '--use_tl', action='store_true',
        help='Use TransformerLens model instead of HuggingFace model')
    parser.add_argument(
        '--is_test', action='store_true')
    parser.add_argument(
        '--prompt_name', default='all')

    args = parser.parse_args()

    print(timestamp(), 'Begin loading model')
    model = load_model(
        args.model, device=args.device,
        use_hf=not args.use_tl,
        dtype=torch.float16 if args.device.startswith(
            "cuda") else torch.float32
    )
    print(timestamp(), 'Finished loading model')

    model_family = get_model_family(args.model)
    torch.set_grad_enabled(False)

    if args.prompt_name == 'all':
        prompt_list = list(ENTITY_PROMPTS[args.entity_type].keys())
    else:
        prompt_list = [args.prompt_name]

    for prompt_name in prompt_list:
        tokenized_dataset = common.load_tokenized_dataset(
            args.entity_type, prompt_name, model_family)

        if args.is_test:
            tokenized_dataset = tokenized_dataset.select(range(10))

        activation_save_path = os.path.join(
            os.getenv('ACTIVATION_DATASET_DIR', 'activation_datasets'),
            args.model,
            args.entity_type
        )
        os.makedirs(activation_save_path, exist_ok=True)

        print(timestamp(),
              f'Begin processing {args.model} {args.entity_type} {prompt_name}')

        if args.use_tl:
            layer_activations = get_layer_activations_tl(
                args, model, tokenized_dataset,
                device=args.device,
            )
        else:
            layer_activations = get_layer_activations_hf(
                args, model, tokenized_dataset,
                device=args.device,
            )

        for layer_ix, activations in layer_activations.items():
            save_name = f'{args.entity_type}.{args.activation_aggregation}.{prompt_name}.{layer_ix}.pt'
            save_path = os.path.join(activation_save_path, save_name)
            activations = adjust_precision(
                activations.to(torch.float32), args.save_precision, per_channel=True)
            torch.save(activations, save_path)
