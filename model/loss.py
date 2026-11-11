import torch

def compute_joint(view1, view2):
    """Construct a non-negative, symmetric joint probability matrix."""
    if view1.ndim != 2 or view2.ndim != 2:
        raise ValueError("view1 and view2 must be two-dimensional tensors")
    if view1.shape != view2.shape:
        raise ValueError("view1 and view2 must have the same shape")

    batch_size, _ = view1.size()
    if batch_size == 0:
        raise ValueError("view1 and view2 must contain at least one sample")

    probability1 = torch.softmax(view1, dim=1)
    probability2 = torch.softmax(view2, dim=1)
    p_i_j = (
        probability1.t().matmul(probability2)
        + probability2.t().matmul(probability1)
    ) / (2.0 * batch_size)

    eps = torch.finfo(p_i_j.dtype).eps
    p_i_j = p_i_j.clamp_min(eps)
    p_i_j = p_i_j / p_i_j.sum()
    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=9.0):
    """Contrastive loss for maximizing cross-view consistency."""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    eps = torch.finfo(p_i_j.dtype).eps
    p_i_j = p_i_j.clamp_min(eps)
    p_j = p_j.clamp_min(eps)
    p_i = p_i.clamp_min(eps)

    loss = - p_i_j * (torch.log(p_i_j) - (lamb + 1) * torch.log(p_j) - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss



def loss_metric(latent, attn):
    """
    Metric learning loss

    Parameters
    ------
    latent
        latent features
    attn
        spatial prior graph

    Returns
    ------
    loss
        metric learning loss
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    attn = attn.toarray()
    attn = torch.FloatTensor(attn).to(device)
    dist = torch.cdist(latent, latent, p=2) ** 2
    loss = torch.mean(torch.sum(attn * dist, dim=-1))

    return loss
