import os
from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from env import (
    TwoSpiesEnv,
    K_TURNS, MAX_ACTIONS, MAX_MOVES_SINCE_SEEN,
    NUM_ACTION_IDS, NUM_TECHS, CLOSING_STATES,
)
from net import ActorCritic

INTEL_SCALE = 50.0   # rescales intel to roughly unit range; tune by inspection


def obs_to_tensor(obs: dict, n_cities: int) -> torch.Tensor:
    def one_hot(idx, k):
        return F.one_hot(torch.as_tensor(idx, dtype=torch.long), k).float()

    parts = [
        # Categoricals → one-hot
        one_hot(obs["my_pos"],                    n_cities),
        one_hot(obs["last_seen_opponent"],        n_cities),
        one_hot(obs["opp_moves_since_last_seen"], MAX_MOVES_SINCE_SEEN + 1),
        one_hot(obs["actions_left"],              MAX_ACTIONS + 1),

        # Action histories: (K, M) matrix of action IDs → flatten one-hots
        one_hot(torch.as_tensor(obs["my_action_history"]).long(),  NUM_ACTION_IDS).flatten(),
        one_hot(torch.as_tensor(obs["opp_action_history"]).long(), NUM_ACTION_IDS).flatten(),

        # Bit vectors → float cast
        torch.as_tensor(obs["opp_unlocked_techs"], dtype=torch.float32),
        torch.as_tensor(obs["controlled_by_me"],   dtype=torch.float32),
        torch.as_tensor(obs["controlled_by_opp"],  dtype=torch.float32),
        torch.as_tensor(obs["is_large"],           dtype=torch.float32),

        # Per-city closing state: int per city in {0,1,2,3} → one-hot per city, flatten
        one_hot(torch.as_tensor(obs["moves_till_closing"]).long(), CLOSING_STATES).flatten(),

        # Unbounded scalars: ALWAYS scale these
        torch.tensor([float(obs["my_intel"]) / INTEL_SCALE], dtype=torch.float32),
        torch.as_tensor(obs["amount_of_intel"], dtype=torch.float32) / INTEL_SCALE,
    ]
    return torch.cat(parts)


def obs_dim(n_cities: int) -> int:
    return (
        n_cities                                       # my_pos
        + n_cities                                     # last_seen_opponent
        + (MAX_MOVES_SINCE_SEEN + 1)                   # moves_since_last_seen
        + (MAX_ACTIONS + 1)                            # actions_left
        + K_TURNS * MAX_ACTIONS * NUM_ACTION_IDS       # my_action_history
        + K_TURNS * MAX_ACTIONS * NUM_ACTION_IDS       # opp_action_history
        + NUM_TECHS                                    # opp_unlocked_techs
        + n_cities                                     # controlled_by_me
        + n_cities                                     # controlled_by_opp
        + n_cities                                     # is_large
        + n_cities * CLOSING_STATES                    # moves_till_closing
        + 1                                            # my_intel
        + n_cities                                     # amount_of_intel
    )


# ---- Hyperparameters ----
NUM_ROLLOUTS       = 10000
NUM_STEPS          = 4096    # total per rollout
EPOCHS_PER_ROLLOUT = 4
BATCH_SIZE         = 256
LR                 = 3e-4
GAMMA              = 0.99
LAMBDA_GAE         = 0.95
CLIP_EPS           = 0.2
VALUE_COEF         = 0.5
ENTROPY_COEF       = 0.01


env = TwoSpiesEnv("maps/roundabout.json", world_events=True)
n_cities  = env.num_cities
n_actions = env.action_space.n
model     = ActorCritic(obs_dim(n_cities), n_actions, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


wandb.init(
    project="two-spies",
    config={
        "num_rollouts":       NUM_ROLLOUTS,
        "num_steps":          NUM_STEPS,
        "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
        "batch_size":         BATCH_SIZE,
        "lr":                 LR,
        "gamma":              GAMMA,
        "lambda_gae":         LAMBDA_GAE,
        "clip_eps":           CLIP_EPS,
        "value_coef":         VALUE_COEF,
        "entropy_coef":       ENTROPY_COEF,
        "hidden_dim":         64,
        "map":                "roundabout",
    },
)


pbar = tqdm(range(NUM_ROLLOUTS))
for rollout in pbar:

    # ---- 1. Rollout phase --------------------------------------------------
    trajectories = []                # finalized per-player trajectories
    current = {0: [], 1: []}         # in-progress trajectories per player

    obs, info = env.reset()
    obs  = obs_to_tensor(obs, n_cities)
    mask = torch.as_tensor(info["action_mask"], dtype=torch.bool)

    for step in range(NUM_STEPS):
        actor = env.current_player                              # capture BEFORE step

        with torch.no_grad():
            action, logp, value = model.act(obs, mask)

        next_obs, reward, terminated, truncated, info = env.step(action.item())

        current[actor].append({
            "obs":      obs,
            "action":   action,
            "mask":     mask,
            "logp_old": logp,
            "value":    value,
            "reward":   float(reward),
            "done":     False,
        })

        if terminated or truncated:
            # On STRIKE, the loser also gets a final signal (zero-sum self-play).
            if terminated:
                loser = 1 - actor
                if current[loser]:
                    current[loser][-1]["reward"] -= 1.0
            for p in (0, 1):
                if current[p]:
                    current[p][-1]["done"] = True
                    trajectories.append(current[p])
                    current[p] = []
            obs, info = env.reset()
            obs  = obs_to_tensor(obs, n_cities)
            mask = torch.as_tensor(info["action_mask"], dtype=torch.bool)
        else:
            obs  = obs_to_tensor(next_obs, n_cities)
            mask = torch.as_tensor(info["action_mask"], dtype=torch.bool)

    # Flush in-progress trajectories at rollout boundary.
    for p in (0, 1):
        if current[p]:
            trajectories.append(current[p])

    # ---- 2. GAE + returns (one backward pass per trajectory) ---------------
    for traj in trajectories:
        gae = 0.0
        for t in reversed(range(len(traj))):
            v_next = 0.0 if t == len(traj) - 1 else traj[t + 1]["value"].item()
            delta  = traj[t]["reward"] + GAMMA * v_next - traj[t]["value"].item()
            gae    = delta + GAMMA * LAMBDA_GAE * gae
            traj[t]["advantage"] = gae
            traj[t]["return"]    = gae + traj[t]["value"].item()

    # ---- 3. Flatten into one batch of tensors ------------------------------
    flat = [t for traj in trajectories for t in traj]
    batch = {
        "obs":       torch.stack([t["obs"]      for t in flat]),
        "action":    torch.stack([t["action"]   for t in flat]),
        "mask":      torch.stack([t["mask"]     for t in flat]),
        "logp_old":  torch.stack([t["logp_old"] for t in flat]),
        "advantage": torch.tensor([t["advantage"] for t in flat], dtype=torch.float32),
        "return":    torch.tensor([t["return"]    for t in flat], dtype=torch.float32),
    }

    # Normalize advantages (mean 0, std 1).
    batch["advantage"] = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-8)

    # ---- 4. PPO update phase ----------------------------------------------
    policy_losses, value_losses, entropies, approx_kls, clip_fractions = [], [], [], [], []

    N = batch["obs"].size(0)
    for epoch in range(EPOCHS_PER_ROLLOUT):
        perm = torch.randperm(N)
        for start in range(0, N, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            mb = {k: v[idx] for k, v in batch.items()}

            entropy, logp, value = model.evaluate(mb["obs"], mb["action"], mb["mask"])
            log_ratio = logp - mb["logp_old"]
            ratio     = log_ratio.exp()

            unclipped = ratio * mb["advantage"]
            clipped   = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb["advantage"]
            L_policy  = -torch.min(unclipped, clipped).mean()
            L_value   = (value - mb["return"]).pow(2).mean()
            L_entropy = -entropy.mean()

            loss = L_policy + VALUE_COEF * L_value + ENTROPY_COEF * L_entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Diagnostics (cheap scalar conversions, no graph cost).
            with torch.no_grad():
                policy_losses.append(L_policy.item())
                value_losses.append(L_value.item())
                entropies.append(entropy.mean().item())
                approx_kls.append((-log_ratio).mean().item())
                clip_fractions.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())

    # ---- 5. Log to wandb --------------------------------------------------
    ep_returns = [sum(t["reward"] for t in traj) for traj in trajectories]
    ep_lengths = [len(traj) for traj in trajectories]

    wandb.log({
        "rollout":          rollout,
        "ep_return_mean":   sum(ep_returns) / len(ep_returns) if ep_returns else 0.0,
        "ep_length_mean":   sum(ep_lengths) / len(ep_lengths) if ep_lengths else 0.0,
        "ep_count":         len(trajectories),
        "policy_loss":      sum(policy_losses) / len(policy_losses),
        "value_loss":       sum(value_losses)  / len(value_losses),
        "entropy":          sum(entropies)     / len(entropies),
        "approx_kl":        sum(approx_kls)    / len(approx_kls),
        "clip_fraction":    sum(clip_fractions) / len(clip_fractions),
        "advantage_std":    batch["advantage"].std().item(),
    })

    pbar.set_postfix(
        ret=f"{(sum(ep_returns) / len(ep_returns)) if ep_returns else 0.0:+.3f}",
        pi=f"{sum(policy_losses) / len(policy_losses):+.3f}",
        v=f"{sum(value_losses) / len(value_losses):.3f}",
        H=f"{sum(entropies) / len(entropies):.3f}",
    )

# ---- Save final checkpoint ---------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
ckpt_name = wandb.run.name or datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_path = f"checkpoints/{ckpt_name}.pt"
torch.save(model.state_dict(), ckpt_path)
print(f"Saved checkpoint to {ckpt_path}")

wandb.finish()
