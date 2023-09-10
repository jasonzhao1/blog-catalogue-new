import torch
import torch.nn.functional as F
from tqdm import tqdm


# These are copied directly from the paper; need to be further refined

def pre_residual_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for residual correlation"""
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    y[label_idx] = F.one_hot(labels[label_idx],c).float().squeeze(1) - model_out[label_idx]
    return y


def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, device='cuda', display=True):
    """general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation"""
    adj = adj.to(device)
    orig_device = y.device
    y = y.to(device)
    result = y.clone()
    for _ in tqdm(range(num_propagations), disable = not display):
        result = alpha * (adj @ result)
        if alpha_term:
            result += (1-alpha)*y
        else:
            result += y
        result = post_step(result)
    return result.to(orig_device)


def pre_outcome_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for outcome correlation"""

    labels = labels.cpu()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    y = model_out.clone()
    if len(label_idx) > 0:
        y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1)

    return y


def double_correlation_autoscale(data, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2,
                                 num_propagations2, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    if train_only:
        label_idx = torch.cat([split_idx['train']])
        residual_idx = split_idx['train']
    else:
        label_idx = torch.cat([split_idx['train'], split_idx['valid']])
        residual_idx = label_idx

    y = pre_residual_correlation(labels=data.y.data, model_out=model_out, label_idx=residual_idx)
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1,
                                        post_step=lambda x: torch.clamp(x, -1.0, 1.0), alpha_term=True, display=display,
                                        device=device)

    orig_diff = y[residual_idx].abs().sum() / residual_idx.shape[0]
    resid_scale = (orig_diff / resid.abs().sum(dim=1, keepdim=True))
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = (resid_scale > 1000)
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale * resid
    res_result[res_result.isnan()] = model_out[res_result.isnan()]
    y = pre_outcome_correlation(labels=data.y.data, model_out=res_result, label_idx=label_idx)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2,
                                         post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True, display=display,
                                         device=device)

    return res_result, result


# TODO: finish the importing between functions
def get_run_from_file(out):
    return int(os.path.splitext(os.path.basename(out))[0])


def model_load(file, device='cpu'):
    result = torch.load(file, map_location='cpu')
    # TODO
    run = get_run_from_file(file)
    try:
        split = torch.load(f'{file}.split', map_location='cpu')
    except:
        split = None

    mx_diff = (result.sum(dim=-1) - 1).abs().max()
    if mx_diff > 1e-1:
        print(f'Max difference: {mx_diff}')
        print("model output doesn't seem to sum to 1. Did you remember to exp() if your model outputs log_softmax()?")
        raise Exception
    if split is not None:
        return (result, split), run
    else:
        return result, run


def evaluate_params(data, eval_test, model_outs, split_idx, params, fn=double_correlation_autoscale):
    logger = SimpleLogger('evaluate params', [], 2)

    for out in model_outs:
        model_out, run = model_load(out)
        if isinstance(model_out, tuple):
            model_out, t = model_out
            split_idx = t
        res_result, result = fn(data, model_out, split_idx, **params)
        valid_acc, test_acc = eval_test(result, split_idx['valid']), eval_test(result, split_idx['test'])
        print(f"Valid: {valid_acc}, Test: {test_acc}")
        logger.add_result(run, (), (valid_acc, test_acc))
    print('Valid acc -> Test acc')
    logger.display()
    return logger