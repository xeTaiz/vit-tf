import torch
import torch.nn.functional as F

def sharpen(p, T):
    sharp_p = p**(1./T)
    return sharp_p / torch.sum(sharp_p, dim=1, keepdim=True)

def snn(query, support, labels, tau=0.1):
    q = F.normalize(query, dim=1)      # (BS, F)
    s = F.normalize(support, dim=1)    # (M, F)
    # (BS, F) @ (F, M) -> (BS, M)
    # (BS, M) @ (M, C) -> (BS, C)
    return F.softmax(q @ s.T / tau, dim=1) @ labels

def paws_loss(anchor_views, anchor_supports, anchor_labels,
              target_views, target_supports, target_labels, clas_pred=None, temperature=0.25, tau=0.1):
    # Get Anchor pseudo-labels
    probs = snn(anchor_views, anchor_supports, anchor_labels, tau=tau)
    # Get positive pseudo-labels
    with torch.no_grad():
        targs = sharpen(snn(target_views, target_supports, target_labels, tau=tau), T=temperature)
        targs[targs < 1e-4] *= 0
    # Cross-Entropy loss H(targets, queries)
    loss = torch.mean(torch.sum(torch.log(probs**(-targs)), dim=1))

    # Mean-Entropy maximization regularization
    avg_probs = torch.mean(sharpen(probs, T=temperature), dim=0)
    memax_loss = -torch.sum(torch.log(avg_probs**(-avg_probs)))
    if clas_pred is not None:
        clas_targ = torch.cat([anchor_labels, sharpen(probs.detach(), T=temperature)], dim=0)
        clas_loss = F.cross_entropy(clas_pred, clas_targ)
        return loss, memax_loss, clas_loss
    else:
        return loss, memax_loss

def transform_paws_crops(crops, noise_std=0.05, flip=True, permute=True):
    if noise_std > 0.0:
        anchors = crops + torch.randn_like(crops) * noise_std
        positiv = crops + torch.randn_like(crops) * noise_std
    else:
        anchors, positiv = crops, crops.clone()
    if permute:
        permutations = [  # All Possible permutations for volume
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
        ]
        idx_anc, idx_pos = torch.randint(0, len(permutations), (2,)).tolist()
        pre_shap = tuple(range(crops.ndim - 3))
        post_shap_anc = tuple(map(lambda d: d + len(pre_shap), permutations[idx_anc]))
        post_shap_pos = tuple(map(lambda d: d + len(pre_shap), permutations[idx_pos]))
        anchors = anchors.permute(*pre_shap, *post_shap_anc)
        positiv = anchors.permute(*pre_shap, *post_shap_pos)
    if flip:
        flips = torch.rand(6) < 0.5
        anchors = anchors.flip(dims=[i for i,f in enumerate(flips.tolist()[:3]) if f])
        positiv = positiv.flip(dims=[i for i,f in enumerate(flips.tolist()[3:]) if f])

    return torch.cat([anchors, positiv], dim=0)
