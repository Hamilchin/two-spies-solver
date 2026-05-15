"""Eval: pit two checkpoints against each other in self-play and view the games.

Usage:
    python eval.py checkpoints/foo.pt checkpoints/bar.pt
    python eval.py checkpoints/foo.pt checkpoints/bar.pt --episodes 5
    python eval.py checkpoints/foo.pt checkpoints/bar.pt --map maps/roundabout.json
"""
from __future__ import annotations
import argparse

import torch

from env import TwoSpiesEnv
from net import ActorCritic
from play import replay
from train import obs_to_tensor, obs_dim


HIDDEN_DIM = 64   # must match how the checkpoints were trained


def load_model(path: str, obs_d: int, n_actions: int) -> ActorCritic:
    model = ActorCritic(obs_d, n_actions, HIDDEN_DIM)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def play_episode(env: TwoSpiesEnv, model_p1: ActorCritic, model_p2: ActorCritic) -> list[dict]:
    """One self-play episode with model_p1 as P1 and model_p2 as P2.
    Returns the list of snapshots from reset through termination/truncation."""
    snapshots: list[dict] = []
    obs, info = env.reset()
    snapshots.append(env.snapshot())

    while True:
        obs_t = obs_to_tensor(obs, env.num_cities)
        mask  = torch.as_tensor(info["action_mask"], dtype=torch.bool)
        actor = env.current_player
        model = model_p1 if actor == 0 else model_p2

        with torch.no_grad():
            action, _, _ = model.act(obs_t, mask)

        obs, _, terminated, truncated, info = env.step(action.item())
        snapshots.append(env.snapshot())

        if terminated or truncated:
            return snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("p1_ckpt", help="path to P1 model checkpoint (.pt)")
    parser.add_argument("p2_ckpt", help="path to P2 model checkpoint (.pt)")
    parser.add_argument("--map", default="maps/roundabout.json")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of self-play episodes to record before replay")
    args = parser.parse_args()

    env = TwoSpiesEnv(args.map, world_events=True)
    obs_d = obs_dim(env.num_cities)
    n_actions = env.action_space.n

    p1 = load_model(args.p1_ckpt, obs_d, n_actions)
    p2 = load_model(args.p2_ckpt, obs_d, n_actions)

    snapshots: list[dict] = []
    for i in range(args.episodes):
        snaps = play_episode(env, p1, p2)
        print(f"Episode {i + 1}/{args.episodes}: {len(snaps)} steps")
        snapshots.extend(snaps)

    replay(snapshots, args.map)


if __name__ == "__main__":
    main()
