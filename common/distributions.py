"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import torch as th
from gym import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.preprocessing import get_meta_action_dim


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken meta_action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic meta_action
        """

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely meta_action (deterministic output)
        from the probability distribution

        :return: the stochastic meta_action
        """

    def get_meta_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return meta_actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def meta_actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: meta_actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: meta_actions and log prob
        """


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous meta_actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_meta_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous meta_actions.

    :param meta_action_dim:  Dimension of the meta_action space.
    """

    def __init__(self, meta_action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.meta_action_dim = meta_action_dim
        self.mean_meta_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the meta_action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_meta_actions = nn.Linear(latent_dim, self.meta_action_dim)
        # TODO: allow meta_action dependent std
        log_std = nn.Parameter(th.ones(self.meta_action_dim) * log_std_init, requires_grad=True)
        return mean_meta_actions, log_std

    def proba_distribution(self, mean_meta_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_meta_actions:
        :param log_std:
        :return:
        """
        meta_action_std = th.ones_like(mean_meta_actions) * log_std.exp()
        self.distribution = Normal(mean_meta_actions, meta_action_std)
        return self

    def log_prob(self, meta_actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of meta_actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param meta_actions:
        :return:
        """
        log_prob = self.distribution.log_prob(meta_actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def meta_actions_from_params(self, mean_meta_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_meta_actions, log_std)
        return self.get_meta_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_meta_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an meta_action
        given the distribution parameters.

        :param mean_meta_actions:
        :param log_std:
        :return:
        """
        meta_actions = self.meta_actions_from_params(mean_meta_actions, log_std)
        log_prob = self.log_prob(meta_actions)
        return meta_actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param meta_action_dim: Dimension of the meta_action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, meta_action_dim: int, epsilon: float = 1e-6):
        super(SquashedDiagGaussianDistribution, self).__init__(meta_action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_meta_actions = None

    def proba_distribution(self, mean_meta_actions: th.Tensor, log_std: th.Tensor) -> "SquashedDiagGaussianDistribution":
        super(SquashedDiagGaussianDistribution, self).proba_distribution(mean_meta_actions, log_std)
        return self

    def log_prob(self, meta_actions: th.Tensor, gaussian_meta_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_meta_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_meta_actions = TanhBijector.inverse(meta_actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super(SquashedDiagGaussianDistribution, self).log_prob(gaussian_meta_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - meta_actions ** 2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_meta_actions = super().sample()
        return th.tanh(self.gaussian_meta_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_meta_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_meta_actions)

    def log_prob_from_params(self, mean_meta_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        meta_action = self.meta_actions_from_params(mean_meta_actions, log_std)
        log_prob = self.log_prob(meta_action, self.gaussian_meta_actions)
        return meta_action, log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete meta_actions.

    :param meta_action_dim: Number of discrete meta_actions
    """

    def __init__(self, meta_action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.meta_action_dim = meta_action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the meta_action layer)
        :return:
        """
        meta_action_logits = nn.Linear(latent_dim, self.meta_action_dim)
        return meta_action_logits

    def proba_distribution(self, meta_action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=meta_action_logits)
        return self

    def log_prob(self, meta_actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(meta_actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def meta_actions_from_params(self, meta_action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(meta_action_logits)
        return self.get_meta_actions(deterministic=deterministic)

    def log_prob_from_params(self, meta_action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        meta_actions = self.meta_actions_from_params(meta_action_logits)
        log_prob = self.log_prob(meta_actions)
        return meta_actions, log_prob


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete meta_actions.

    :param meta_action_dims: List of sizes of discrete meta_action spaces
    """

    def __init__(self, meta_action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.meta_action_dims = meta_action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the meta_action layer)
        :return:
        """

        meta_action_logits = nn.Linear(latent_dim, sum(self.meta_action_dims))
        return meta_action_logits

    def proba_distribution(self, meta_action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(logits=split) for split in th.split(meta_action_logits, tuple(self.meta_action_dims), dim=1)]
        return self

    def log_prob(self, meta_actions: th.Tensor) -> th.Tensor:
        # Extract each discrete meta_action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(meta_action) for dist, meta_action in zip(self.distribution, th.unbind(meta_actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def meta_actions_from_params(self, meta_action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(meta_action_logits)
        return self.get_meta_actions(deterministic=deterministic)

    def log_prob_from_params(self, meta_action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        meta_actions = self.meta_actions_from_params(meta_action_logits)
        log_prob = self.log_prob(meta_actions)
        return meta_actions, log_prob


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary meta_action spaces.

    :param meta_action_dim: Number of binary meta_actions
    """

    def __init__(self, meta_action_dims: int):
        super(BernoulliDistribution, self).__init__()
        self.meta_action_dims = meta_action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the meta_action layer)
        :return:
        """
        meta_action_logits = nn.Linear(latent_dim, self.meta_action_dims)
        return meta_action_logits

    def proba_distribution(self, meta_action_logits: th.Tensor) -> "BernoulliDistribution":
        self.distribution = Bernoulli(logits=meta_action_logits)
        return self

    def log_prob(self, meta_actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(meta_actions).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.round(self.distribution.probs)

    def meta_actions_from_params(self, meta_action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(meta_action_logits)
        return self.get_meta_actions(deterministic=deterministic)

    def log_prob_from_params(self, meta_action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        meta_actions = self.meta_actions_from_params(meta_action_logits)
        log_prob = self.log_prob(meta_actions)
        return meta_actions, log_prob


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an meta_action with that noise.

    :param meta_action_dim: Dimension of the meta_action space.
    :param full_std: Whether to use (n_features x n_meta_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        meta_action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
    ):
        super(StateDependentNoiseDistribution, self).__init__()
        self.meta_action_dim = meta_action_dim
        self.latent_sde_dim = None
        self.mean_meta_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: th.Tensor) -> th.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = th.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = th.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return th.ones(self.latent_sde_dim, self.meta_action_dim).to(log_std.device) * std

    def sample_weights(self, log_std: th.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = -2.0, latent_sde_dim: Optional[int] = None
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic meta_action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the meta_action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        # Network for the deterministic meta_action, it represents the mean of the distribution
        mean_meta_actions_net = nn.Linear(latent_dim, self.meta_action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = th.ones(self.latent_sde_dim, self.meta_action_dim) if self.full_std else th.ones(self.latent_sde_dim, 1)
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_meta_actions_net, log_std

    def proba_distribution(
        self, mean_meta_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> "StateDependentNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_meta_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_meta_actions, th.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, meta_actions: th.Tensor) -> th.Tensor:
        if self.bijector is not None:
            gaussian_meta_actions = self.bijector.inverse(meta_actions)
        else:
            gaussian_meta_actions = meta_actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_meta_actions)
        # Sum along meta_action dim
        log_prob = sum_independent_dims(log_prob)

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= th.sum(self.bijector.log_prob_correction(gaussian_meta_actions), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        if self.bijector is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        meta_actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(meta_actions)
        return meta_actions

    def mode(self) -> th.Tensor:
        meta_actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(meta_actions)
        return meta_actions

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_meta_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def meta_actions_from_params(
        self, mean_meta_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_meta_actions, log_std, latent_sde)
        return self.get_meta_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_meta_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        meta_actions = self.meta_actions_from_params(mean_meta_actions, log_std, latent_sde)
        log_prob = self.log_prob(meta_actions)
        return meta_actions, log_prob


class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the meta_action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


def make_proba_distribution(
    meta_action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of meta_action space

    :param meta_action_space: the input meta_action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(meta_action_space, spaces.Box):
        assert len(meta_action_space.shape) == 1, "Error: the meta_action space must be a vector"
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_meta_action_dim(meta_action_space), **dist_kwargs)
    elif isinstance(meta_action_space, spaces.Discrete):
        return CategoricalDistribution(meta_action_space.n, **dist_kwargs)
    elif isinstance(meta_action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(meta_action_space.nvec, **dist_kwargs)
    elif isinstance(meta_action_space, spaces.MultiBinary):
        return BernoulliDistribution(meta_action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for meta_action space"
            f"of type {type(meta_action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


def kl_divergence(dist_true: Distribution, dist_pred: Distribution) -> th.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence

    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, MultiCategoricalDistribution):
        assert dist_pred.meta_action_dims == dist_true.meta_action_dims, "Error: distributions must have the same input space"
        return th.stack(
            [th.distributions.kl_divergence(p, q) for p, q in zip(dist_true.distribution, dist_pred.distribution)],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    else:
        return th.distributions.kl_divergence(dist_true.distribution, dist_pred.distribution)
