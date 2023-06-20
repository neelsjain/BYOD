import torch.nn.functional as F
import torch


def JSD(prob_dist_1, prob_dist_2):
    """
    Calcaulates the symmetric KL divergence between the two distributions of probabilities -- 0.5*(KL(a||M) + KL(b||M))

    Written to handle batches (i.e batch x prob_distribution).

    Returns batch x score
    """
    ref_dist = 0.5 * (prob_dist_1 + prob_dist_2)
    KL_1 = F.kl_div(torch.log(ref_dist), prob_dist_1, reduction="none").sum(dim=-1)
    KL_2 = F.kl_div(torch.log(ref_dist), prob_dist_2, reduction="none").sum(dim=-1)

    return 0.5 * (KL_1 + KL_2)
