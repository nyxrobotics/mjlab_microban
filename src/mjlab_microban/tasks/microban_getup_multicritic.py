"""Multi-critic PPO for get-up: one value head per HoST-style reward group.

HoST (arXiv:2502.08378) names and fixes the exact failure mode measured on
this task (reaches/overshoots standing height fine -- mean max head height
113-125% of target across three separate runs -- but sustains it in only
1-3/16 rollout envs, unchanged across ~10000+ additional iterations and
several reward-weight experiments): their own ablation shows a SINGLE critic
trying to explain return variance across many competing reward terms
collapses success to zero on their task. Splitting into separate value heads
per reward GROUP (task/style/regu/post), each advantage normalized
independently before being linearly combined into one policy-gradient
signal, is what actually fixes it -- not any single reward term's weight.

rsl_rl (this project's RL library) has no built-in multi-critic support, so
this subclasses only what's needed for a feedforward (non-recurrent),
single-GPU, no-RND, no-symmetry PPO run -- get-up's own configuration.
Everything else (actor, the policy surrogate loss, KL-adaptive LR, the
value-loss shape) is inherited from rsl_rl's PPO/RolloutStorage unchanged:
torch's elementwise broadcasting already generalizes correctly to the extra
trailing reward-group dimension almost everywhere. Only reward sourcing
(group sums instead of one scalar), the rollout buffer shapes, and the
final advantage combination needed deliberate reshaping.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

# HoST-style grouping (their task/style/regu/post) mapped onto this task's
# own reward terms. Every term active in microban_getup_env_cfg.py must
# appear in exactly one group -- construct_algorithm below raises loudly if
# any term is missing or any listed name isn't actually active, so this
# can't silently drift out of sync with that file.
GETUP_REWARD_GROUPS: dict[str, tuple[str, ...]] = {
    "task": ("head_height", "head_height_sq", "on_feet"),
    "style": (
        "dof_pos_limits",
        "self_collisions",
        "raw_target_clip_excess",
        "hands_released",
    ),
    "regu": ("action_rate_l2", "joint_torques_l2"),
    "post": (
        "standing_bonus",
        "standing_stability",
        "upright_balance",
        "balance_recovery",
        "standing_torque",
        "home_stillness",
        "standing_pose",
        "hip_roll_pose",
    ),
}
GETUP_REWARD_GROUP_NAMES: tuple[str, ...] = tuple(GETUP_REWARD_GROUPS)
# HoST's own reward-level group coefficients (r = 2.5*task + 1*style + 0.1*regu
# + 1*post), reused here as the advantage-COMBINATION weights. Normalizing
# every group to unit variance before summing (see compute_returns) would
# otherwise erase this task's own deliberately-tuned relative reward-term
# weights (e.g. action_rate_l2 at -0.02 is meant to be a light regularizer,
# not an equal partner to head_height at 36.0), so weighting the normalized
# groups unevenly (rather than a flat 1:1:1:1 sum) seemed worth doing on
# principle -- though a direct 50-iteration/256-env smoke-test comparison
# against equal weighting showed both decline nearly identically over that
# short window, and a same-settings run of the ORIGINAL single-critic PPO
# declines the same way too (+6 -> -325): this is just this task's normal
# early-training shape at this tiny scale (sparse success terms haven't
# fired yet, only always-on penalties have accumulated), not something these
# coefficients fix or equal weighting breaks. Kept as the more principled,
# paper-grounded default anyway, not because it was measured to matter here.
GETUP_REWARD_GROUP_COEFFS: dict[str, float] = {
    "task": 2.5,
    "style": 1.0,
    "regu": 0.1,
    "post": 1.0,
}
NUM_REWARD_GROUPS = len(GETUP_REWARD_GROUP_NAMES)


class MultiCriticRolloutStorage(RolloutStorage):
    """RolloutStorage with values/returns/rewards shaped (T, N, K), not (T, N, 1).

    advantages stays (T, N, 1): it holds the already-combined, single scalar
    used by the (unmodified) PPO policy surrogate loss -- see
    MultiCriticPPO.compute_returns for where the K group advantages become
    that one column.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        if training_type != "rl":
            raise ValueError("MultiCriticRolloutStorage only supports rl training")
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        k = NUM_REWARD_GROUPS
        t, n = num_transitions_per_env, num_envs
        self.rewards = torch.zeros(t, n, k, device=self.device)
        self.values = torch.zeros(t, n, k, device=self.device)
        self.returns = torch.zeros(t, n, k, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards)  # (N, K) -- no .view(-1, 1) here, unlike the base class
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)  # type: ignore
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                for p in transition.distribution_params  # type: ignore
            )
        for i, p in enumerate(transition.distribution_params):  # type: ignore
            self.distribution_params[i][self.step].copy_(p)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1


class MultiCriticPPO(PPO):
    """PPO with one value head per HoST-style reward group.

    See this module's own docstring for the full rationale.
    """

    def __init__(
        self,
        actor,
        critic,
        storage: MultiCriticRolloutStorage,
        *,
        env: VecEnv,
        term_to_group: torch.Tensor,
        **kwargs,
    ) -> None:
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("MultiCriticPPO does not support RND or symmetry")
        super().__init__(actor, critic, storage, **kwargs)
        self._env = env
        self._term_to_group = term_to_group.to(self.device)
        self._group_coeffs = torch.tensor(
            [GETUP_REWARD_GROUP_COEFFS[name] for name in GETUP_REWARD_GROUP_NAMES],
            device=self.device,
        )

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        del rewards  # The single scalar sum isn't used -- group rewards are recomputed below.
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        unwrapped = self._env.unwrapped
        # RewardManager already computes this every step for Episode_Reward/*
        # logging (mjlab/managers/reward_manager.py); it's the unscaled
        # (pre-dt) weighted value per term, so multiply by step_dt to match
        # what the single scalar reward_buf would have summed to.
        step_reward = unwrapped.reward_manager._step_reward * unwrapped.step_dt
        group_rewards = torch.zeros(step_reward.shape[0], NUM_REWARD_GROUPS, device=step_reward.device)
        group_rewards.index_add_(1, self._term_to_group, step_reward)
        self.transition.rewards = group_rewards.to(self.device)
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards = self.transition.rewards + self.gamma * self.transition.values * time_outs

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Per-group GAE, each advantage column normalized independently,
        weighted by GETUP_REWARD_GROUP_COEFFS, then summed into the one
        combined column the (unmodified) PPO policy surrogate loss reads --
        HoST's "advantages normalized then linearly combined". Returns/
        values stay per-group (T, N, K): the (unmodified) PPO value loss
        already reduces over every element, which is exactly the sum of each
        head's own squared error against its own group's return.
        """
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        raw_advantage = st.returns - st.values  # (T, N, K)
        normalized = (raw_advantage - raw_advantage.mean(dim=(0, 1), keepdim=True)) / (
            raw_advantage.std(dim=(0, 1), keepdim=True) + 1e-8
        )
        st.advantages = (normalized * self._group_coeffs).sum(dim=-1, keepdim=True)  # (T, N, 1)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "MultiCriticPPO":
        alg_class: type[MultiCriticPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic = critic_class(obs, cfg["obs_groups"], "critic", NUM_REWARD_GROUPS, **cfg["critic"]).to(device)
        print(f"Critic Model ({NUM_REWARD_GROUPS} reward-group heads: {GETUP_REWARD_GROUP_NAMES}): {critic}")

        storage = MultiCriticRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        active_terms = list(env.unwrapped.reward_manager.active_terms)
        name_to_group = {
            name: group_idx
            for group_idx, names in enumerate(GETUP_REWARD_GROUPS.values())
            for name in names
        }
        missing = [name for name in active_terms if name not in name_to_group]
        if missing:
            raise ValueError(f"Reward term(s) {missing} not assigned to a GETUP_REWARD_GROUPS group")
        extra = [name for name in name_to_group if name not in active_terms]
        if extra:
            raise ValueError(f"GETUP_REWARD_GROUPS names {extra} are not active reward terms")
        term_to_group = torch.tensor([name_to_group[name] for name in active_terms], dtype=torch.long)

        alg: MultiCriticPPO = alg_class(
            actor,
            critic,
            storage,
            env=env,
            term_to_group=term_to_group,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **cfg["algorithm"],
        )
        return alg
