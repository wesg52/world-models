import torch
import einops
from utils import adjust_precision


def get_attention_composition(model, direction):
    # TODO: add bias terms

    direction = direction / torch.norm(direction)

    Q_comps, K_comps, V_comps, O_comps = [], [], [], []
    for layer in range(model.cfg.n_layers):
        W_OV = model.OV[layer].T.AB
        W_OV /= torch.norm(W_OV, dim=(1, 2), keepdim=True)

        V_comp = torch.norm(W_OV @ direction, dim=-1)
        O_comp = torch.norm(torch.swapdims(W_OV, 1, 2) @ direction, dim=-1)
        V_comps.append(V_comp)
        O_comps.append(O_comp)

        W_QK = model.QK[layer].T.AB
        W_QK /= torch.norm(W_QK, dim=(1, 2), keepdim=True)

        K_comp = torch.norm(W_QK @ direction, dim=-1)
        Q_comp = torch.norm(torch.swapdims(W_QK, 1, 2) @ direction, dim=-1)
        K_comps.append(K_comp)
        Q_comps.append(Q_comp)

    Q_comp = torch.stack(Q_comps, dim=0)
    K_comp = torch.stack(K_comps, dim=0)
    V_comp = torch.stack(V_comps, dim=0)
    O_comp = torch.stack(O_comps, dim=0)
    return Q_comp, K_comp, V_comp, O_comp


def evaluate_probe_composition(model, direction):
    direction = direction / torch.norm(direction)

    # reshape
    layers, d_model, d_mlp = model.W_in.shape
    W_ins = einops.rearrange(model.W_in, 'l d n -> (l n) d').to(torch.float32)
    W_ins /= torch.norm(W_ins, dim=1, keepdim=True)
    in_sim = (W_ins @ direction).reshape(layers, d_mlp)
    del W_ins

    W_outs = einops.rearrange(
        model.W_out, 'l n d -> (l n) d').to(torch.float32)
    W_outs /= torch.norm(W_outs, dim=1, keepdim=True)
    out_sim = (W_outs @ direction).reshape(layers, d_mlp)
    del W_outs

    # W_E is (d_vocab, d_model), W_U is (d_model, d_vocab)
    W_E = model.W_E / torch.norm(model.W_E, dim=-1, keepdim=True)
    W_U = model.W_U / torch.norm(model.W_U, dim=0, keepdim=True)

    W_E_sim = W_E @ direction
    W_U_sim = W_U.T @ direction

    del W_E, W_U

    Q_comp, K_comp, V_comp, O_comp = get_attention_composition(
        model, direction)

    # adjust precision and return
    composition = {
        'W_in': adjust_precision(in_sim, per_channel=False, cos_sim=True),
        'W_out': adjust_precision(out_sim, per_channel=False, cos_sim=True),
        'W_E': adjust_precision(W_E_sim, per_channel=False, cos_sim=True),
        'W_U': adjust_precision(W_U_sim, per_channel=False, cos_sim=True),
        'o_comp': adjust_precision(O_comp, per_channel=False, cos_sim=True),
        'v_comp': adjust_precision(V_comp, per_channel=False, cos_sim=True),
        'q_comp': adjust_precision(Q_comp, per_channel=False, cos_sim=True),
        'k_comp': adjust_precision(K_comp, per_channel=False, cos_sim=True),
    }
    return composition
