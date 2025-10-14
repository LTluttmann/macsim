import abc
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from macsim.utils.ops import gather_by_index

# class Loss(abc.ABC):

#     def __init__(self, eval_per_agent):
#         self.eval_per_agent = eval_per_agent

#     def _logits_to_logp(self, logits: torch.Tensor, mask: torch.Tensor):
#         if torch.is_grad_enabled() and self.eval_per_agent:
#             # when training we evaluate on a per agent basis
#             # perform softmax per agent
#             logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
#             # flatten logp for selection
#             flat_logp = rearrange(logp, "b m j -> b (m j)")

#         else:
#             # when rolling out, we sample iteratively from flattened prob dist
#             flat_logp = self.dec_strategy.logits_to_logp(logits=logits.flatten(1,2), mask=mask.flatten(1,2))
            
#         return flat_logp
    

#     @abc.abstractmethod
#     def __call__(self, logits, td, env):
#         pass



# class CrossEntropy(Loss):




#     def __call__(self, logits, td, env):



def ce_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        td: TensorDict,
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        **kwargs
    ):
    """standard cross entropy loss"""
    assert logp.dim() == entropy.dim(), f"Logprobs and Entropy shapes dont match: {logp.shape} vs. {entropy.shape}"
    
    bs = logp.size(0)
    if mask is not None:
        logp[mask] = 0
        entropy[mask] = 0

    # add entropy penalty 
    loss = torch.clamp(logp + entropy_coef * entropy, max=0)

    return -loss.view(bs, -1).sum(1)


def kl_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        td: TensorDict,
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        **kwargs
    ):
    """KL divergence between the target distribution where an expert action is 1/M 
    (with M the number of agents) and 0 else. In this case, we can simply scale the
    CE loss by M, since sum_i p_i (log (q_i)) = p sum_i log(q_i) if p=p_i for all i
    """
    assert logp.dim() == entropy.dim(), f"Logprobs and Entropy shapes dont match: {logp.shape} vs. {entropy.shape}"
    bs, num_agents = logp.shape
    # Uniform target probabilities over agent actions
    target_probs = torch.full_like(logp, 1.0 / num_agents)
    if mask is not None:
        target_probs[mask] = 0
        entropy[mask] = 0  # for masked entries, simply add no penalty
    # KL divergence: sum_i p(i) * (log(p(i)) - log(q(i)))
    log_target = torch.log(target_probs)
    kl_div = torch.sum(target_probs * (log_target - logp), dim=-1)
    loss = kl_div - entropy_coef * entropy
    return loss


def simple_listnet_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        td: TensorDict,
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        **kwargs
    ):
    """This ListNet assumes uniform distribution of the 'correct' agent action, i.e. the target distribution
    for an expert action is 1/M (with M the number of agents) and 0 else. In this case, we can simply scale the
    CE loss by M, since sum_i p_i (log (q_i)) = p sum_i log(q_i) if p=p_i for all i
    """
    assert logp.dim() == entropy.dim(), f"Logprobs and Entropy shapes dont match: {logp.shape} vs. {entropy.shape}"
    bs = logp.size(0)
    if mask is not None:
        logp[mask] = 0
        entropy[mask] = 0
        denom = mask.view(bs, -1).logical_not().sum(1) + 1e-6

    # add entropy penalty 
    loss = torch.clamp(logp + entropy_coef * entropy, max=0)

    if mask is not None:
        loss = loss.view(bs, -1).sum(1) / denom
    else:
        loss = loss.view(bs, -1).mean(1)

    return -loss


def listnet_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        td: TensorDict,
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        alpha: float = 0.1,
        **kwargs
    ):
    """ListNet inspired loss. This loss assumes that the logps are ordered corresponding to the
    order the machinens sampled actions during experience collection. The loss enforces a precendence
    by weighting machines that sampled first stronger. This makes intuitive sense, because these agents
    disrupted the sampling space of succeeding machines
    """
    assert logp.dim() == entropy.dim(), f"Logprobs and Entropy shapes dont match: {logp.shape} vs. {entropy.shape}"

    bs, N = logp.shape
    # (bs, num_actions)
    logp = logp.view(bs, -1)
    
    # ranks = torch.arange(1, N+1, device=logp.device).view(1, N).expand_as(logp)
    # weights = torch.exp(-(alpha * (ranks - 1)))
    with torch.no_grad():
        scores = gather_by_index(td["logits"].flatten(-2,-1), td["action"]["idx"], dim=-1)
        # (bs, num_actions)
        target_dist = torch.softmax(scores, dim=-1)
        
    if mask is not None:
        # TODO is this sufficient?
        scores[mask] = float("-inf")
        entropy[mask] = 0  # for masked entries, simply add no penalty

    # (bs, actions)
    ce_loss = -torch.mul(target_dist, logp)
    loss = torch.clamp(ce_loss - entropy_coef * entropy, min=0)
        
    loss = loss.sum(-1)

    return loss


def calc_adv_weights(adv: torch.Tensor, temp: float = 1.0, weight_clip: float = 10.0):

    adv_mean = adv.mean()
    adv_std = adv.std()

    norm_adv = (adv - adv_mean) / (adv_std + 1e-5)

    weights = torch.exp(norm_adv / temp)

    weights = torch.minimum(weights, torch.full_like(weights, fill_value=weight_clip))
    return weights


def pairwise_hinge_loss(logits, labels, margin=1.0):
    # Compute score difference
    diff = logits[:, 0] - logits[:, 1]  # shape (bs,)
    
    # Flip sign if label is 0 (i.e., action 2 is better)
    signed_diff = diff * (2 * labels.float() - 1)
    
    # Apply hinge loss
    loss = torch.clamp(margin - signed_diff, min=0).mean()
    return loss


def listwise_soft_ranking_loss(logits, rewards, temperature=0.0001, tanh=None):
    """
    Computes listwise ranking loss based on reward-derived soft targets.

    Inputs:
        logits: (B, S) - model scores for S actions
        rewards: (B, S) - rewards for each action
        temperature: float - sharpness of reward-to-probability

    Returns:
        Scalar loss averaged over batch
    """
    # Target probabilities from rewards (softmax over rewards)
    with torch.no_grad():
        target_probs = F.softmax(rewards / temperature, dim=-1)  # (B, S)

    # Log-probabilities from logits
    if tanh is not None:
        logits = torch.tanh(logits) * tanh

    log_probs = F.log_softmax(logits, dim=-1)  # (B, S)

    # Cross-entropy between target and predicted distributions
    loss = -(target_probs * log_probs).sum(dim=-1)

    return loss


def multi_target_max_reward_loss(log_probs, rewards, epsilon=1e-6):
    """
    scores: (batch, num_actions)
    rewards: (batch, num_actions)
    epsilon: tolerance for allowing near-max rewards

    Returns: scalar loss
    """
    with torch.no_grad():
        max_rewards = rewards.max(dim=1, keepdim=True).values
        is_optimal = (rewards >= (max_rewards - epsilon)).float()
        target_probs = is_optimal / is_optimal.sum(dim=1, keepdim=True)  # normalize

    loss = -(target_probs * log_probs).sum(dim=1)
    return loss


def soft_preference_pair_loss(logits, rewards, temperature=0.0001):
    """
    Computes pairwise soft preference loss over all action pairs for each sample in the batch.

    Inputs:
        logits: (B, S) - predicted scores for S actions
        rewards: (B, S) - terminal rewards for those actions
        alpha: temperature for reward-to-probability conversion

    Returns:
        Scalar loss averaged over all valid pairs and batch
    """
    assert logits.size(1) == 2, "pairwise loss only for pairs"

    logit_diff = logits[:, 0] - logits[:, 1]
    reward_diff = rewards[:, 0] - rewards[:, 1]
    soft_labels = torch.sigmoid(reward_diff / temperature)
    loss = F.binary_cross_entropy_with_logits(logit_diff, soft_labels, reduction='none')
    return loss 



loss_map = {
    "ce": ce_loss,
    "kl": kl_loss,
    "listnet": simple_listnet_loss,
    "listnet_weighted": listnet_loss
}
