# from typing import Type, Union, Sequence, Optional
# from itertools import pairwise, chain
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Independent, Normal

# SIGMA_MIN = -20
# SIGMA_MAX = 2


# class Actor(nn.Module):
#     def __init__(
#         self,
#         n_actions: int,
#         sigma: float,
#         activation_type: nn.Module = nn.Tanh,
#         input_size: int = 64,
#         hidden_sizes: Sequence[int] = (),
#         max_action: float = 1.0,
#         unbounded: bool = False,
#         conditioned_sigma: bool = False,
#     ) -> None:
#         super().__init__()

#         self.n_actions = n_actions
#         self._max_action = max_action
#         self._unbounded = unbounded
#         self._sigma = sigma

#         self.base_net = [
#             (
#                 nn.Linear(in_dim, out_dim, bias=True),
#                 activation_type()
#             )
#             for in_dim, out_dim in pairwise([input_size] + list(hidden_sizes))
#         ]

#         self.base_net = nn.Sequential(
#             *list(chain.from_iterable(self.base_net))
#         )

#         self.mu = nn.Linear(
#             in_features=hidden_sizes[-1],
#             out_features=self.n_actions,
#             bias=True
#         )

#         self._c_sigma = conditioned_sigma
#         if conditioned_sigma:
#             self.sigma = nn.Linear(
#                 in_features=hidden_sizes[-1],
#                 out_features=self.n_actions,
#                 bias=True
#             )
#         else:
#             self.sigma_param = nn.Parameter(torch.zeros(self.n_actions, 1))

#         self.reset_parameters()

#     def reset_parameters(self, init_range=0.1):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 for param in m.parameters():
#                     param.data.uniform_(-init_range, init_range)

#     def forward(self, obs):
#         logits = self.base_net(obs)
#         mu = self.mu(logits)
#         if not self._unbounded:
#             mu = self._max_action * torch.tanh(mu)
#         # if self._c_sigma:
#         #     sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
#         # else:
#         #     shape = [1] * len(mu.shape)
#         #     shape[1] = -1
#         #     sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu, device=mu.device)).exp()
#         sigma = torch.full_like(mu, fill_value=self._sigma, device=mu.device)
#         return (mu, sigma)


# def dist(*logits):
#     return Independent(Normal(*logits), 1)


# class AverageMetric(object):
#     def __init__(self):
#         self.n = 0
#         self._value = 0.

#     def reset(self) -> None:
#         self.n = 0
#         self._value = 0.

#     def update(self, value) -> None:
#         if torch.is_tensor(value):
#             self.n += value.numel()
#             self._value += value.sum().item()
#         elif isinstance(value, (int, float)):
#             self.n += 1
#             self._value += value
#         else:
#             raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
#                              .format(type(value)))

#     @property
#     def value(self) -> float:
#         if self.n == 0:
#             return 0
#         return self._value / self.n


# class ContinuePolicyGradient:
#     def __init__(
#         self,
#         batch_size: int,
#         input_dim: int,
#         actor: nn.Module,
#         pg_lr: float,
#         milestones,
#         dist_fn: Type[torch.distributions.Distribution] = dist,
#     ) -> None:

#         self.batch_size = batch_size
#         self.input_dim = input_dim

#         self.actor = actor
#         self.dist_fn = dist_fn
#         self.optimizer = optim.Adam(self.actor.parameters(), lr=pg_lr)
#         if milestones is None:
#             self.scheduler = None
#         else:
#             self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

#         self.avg_reward = AverageMetric()

#     def make_decisions(
#         self,
#         eval_mode: bool = False
#     ):
#         observation = torch.zeros((self.batch_size, self.input_dim),
#                                   device=list(self.actor.parameters())[0].device, requires_grad=False)
#         logits = self.actor(observation)

#         dist = self.dist_fn(*logits)
#         if eval_mode:
#             act = logits[0]
#         else:
#             act = dist.sample()

#         sigma = logits[1]
#         return dist, act, sigma

#     def learn(
#         self,
#         dist: torch.distributions.Distribution,
#         actions: torch.Tensor,
#         rewards: torch.Tensor,
#     ):
#         self.avg_reward.update(rewards)
#         normalized_reward = rewards - self.avg_reward.value

#         self.optimizer.zero_grad()
#         loss: torch.Tensor = -(dist.log_prob(actions) * normalized_reward).mean()
#         loss.backward()
#         self.optimizer.step()
#         if self.scheduler is not None:
#             self.scheduler.step()
