from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- Action IDs (used in action_history) ------------------------------------
MOVE, STRIKE, CONTROL, DEEP_COVER, PREP, LOCATE, UNLOCK_RR, UNLOCK_ENC, UNLOCK_SR, WAIT = range(10)
NOOP = 10
HIDDEN = 11
NUM_ACTION_IDS = 12
COSTED_ACTIONS = frozenset({DEEP_COVER, PREP, LOCATE, UNLOCK_RR, UNLOCK_ENC, UNLOCK_SR})

# --- Intel costs ------------------------------------------------------------
INTEL_COST = {DEEP_COVER: 20, PREP: 40, LOCATE: 10}

# --- Tech IDs ---------------------------------------------------------------
TECH_RR, TECH_ENC, TECH_SR = range(3)
NUM_TECHS = 3

# --- Sizing constants -------------------------------------------------------
K_TURNS = 4
MAX_ACTIONS = 4
MAX_MOVES_SINCE_SEEN = 10
# moves_till_closing: 0 = closed, 1 or 2 = closing soon, 3 = not yet marked closing.
NOT_CLOSING = 3
CLOSING_STATES = 4  # values in {0, 1, 2, 3}


class TwoSpiesEnv(gym.Env):
    """Two Spies Gym env.

    Action space layout: Discrete(num_cities + 6), indexed as
        0 .. N-1 :  MOVE to city i
        N        :  WAIT
        N+1      :  CONTROL
        N+2      :  DEEP_COVER
        N+3      :  PREP
        N+4      :  LOCATE
        N+5      :  STRIKE
    `info["action_mask"]` flags which indices are currently legal.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, map_or_path):
        super().__init__()

        # Accept either a JSON dict or a path to a JSON file.
        if isinstance(map_or_path, (str, Path)):
            with Path(map_or_path).open() as f:
                data = json.load(f)
        elif isinstance(map_or_path, dict):
            data = map_or_path
        else:
            raise TypeError("map_or_path must be a dict or a path to a JSON file")

        # Map-derived state.
        self.map_name: str = data["name"]
        self.idx_to_city: list[str] = list(data["cities"])
        self.city_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.idx_to_city)}
        self.num_cities: int = len(self.idx_to_city)

        self.neighbors: list[list[int]] = [[] for _ in range(self.num_cities)]
        for a, b in data["edges"]:
            i, j = self.city_to_idx[a], self.city_to_idx[b]
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        self.large_mask: np.ndarray = np.zeros(self.num_cities, dtype=np.int8)
        for c in data.get("large_cities", []):
            self.large_mask[self.city_to_idx[c]] = 1

        self.start_p1: int = self.city_to_idx[data["start"]["p1"]]
        self.start_p2: int = self.city_to_idx[data["start"]["p2"]]

        n = self.num_cities
        # Action slot offsets.
        self.WAIT_A:       int = n
        self.CONTROL_A:    int = n + 1
        self.DEEP_COVER_A: int = n + 2
        self.PREP_A:       int = n + 3
        self.LOCATE_A:     int = n + 4
        self.STRIKE_A:     int = n + 5
        self.num_actions:  int = n + 6

        # Episode state (populated by reset).
        self.pos: list[int] = [0, 0]
        self.last_seen: list[int] = [0, 0]                # last_seen[viewer] = where viewer last saw the opponent
        self.moves_since_last_seen: list[int] = [0, 0]
        self.intel: list[int] = [0, 0]
        self.unlocked: list[np.ndarray] = [np.zeros(NUM_TECHS, dtype=np.int8)] * 2
        self.deep_cover: list[bool] = [False, False]
        self.pending_extra_actions: list[int] = [0, 0]
        self.current_turn_actions: list[list[int]] = [[], []]
        # Completed turns: each entry is a list of action IDs (any length).
        self.action_history: list[list[list[int]]] = [[], []]
        self.encryption_active_at_turn: list[list[bool]] = [[], []]
        self.current_player: int = 0
        self.actions_left: int = 2
        self.terminated: bool = False

        self.city_states: dict[int, dict] = {}

        # --- Gym spaces ----------------------------------------------------
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict({
            "my_pos":                    spaces.Discrete(n),
            "last_seen_opponent":        spaces.Discrete(n),
            "opp_moves_since_last_seen": spaces.Discrete(MAX_MOVES_SINCE_SEEN + 1),
            "my_action_history":         spaces.Box(0, NUM_ACTION_IDS - 1, (K_TURNS, MAX_ACTIONS), np.int32),
            "opp_action_history":        spaces.Box(0, NUM_ACTION_IDS - 1, (K_TURNS, MAX_ACTIONS), np.int32),
            "opp_unlocked_techs":        spaces.MultiBinary(NUM_TECHS),
            "my_intel":                  spaces.Box(0, np.iinfo(np.int32).max, (), np.int32),
            "actions_left":              spaces.Discrete(MAX_ACTIONS + 1),
            "moves_till_closing":        spaces.MultiDiscrete([CLOSING_STATES] * n),
            "controlled_by_me":          spaces.MultiBinary(n),
            "controlled_by_opp":         spaces.MultiBinary(n),
            "amount_of_intel":           spaces.Box(0, np.iinfo(np.int32).max, (n,), np.int32),
            "is_large":                  spaces.MultiBinary(n),
        })

    # --- Gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.pos = [self.start_p1, self.start_p2]
        self.last_seen = [self.start_p2, self.start_p1]
        self.moves_since_last_seen = [0, 0]
        self.intel = [0, 0]
        self.unlocked = [
            np.zeros(NUM_TECHS, dtype=np.int8),
            np.zeros(NUM_TECHS, dtype=np.int8),
        ]
        self.deep_cover = [False, False]
        self.pending_extra_actions = [0, 0]
        self.current_turn_actions = [[], []]
        self.action_history = [[], []]
        self.encryption_active_at_turn = [[], []]
        self.current_player = 0
        self.actions_left = 2
        self.terminated = False

        self.city_states = {
            i: {
                "moves_till_closing": NOT_CLOSING,
                "controlled_by":      None,   # None | 0 | 1
                "amount_of_intel":    0,
                "is_large":           bool(self.large_mask[i]),
            }
            for i in range(self.num_cities)
        }

        return self.observe(self.current_player), {"action_mask": self._action_mask(self.current_player)}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("step() called on terminated episode")
        if not self._action_mask(self.current_player)[action]:
            raise ValueError(f"illegal action {action} for player {self.current_player}")

        player = self.current_player
        opp = 1 - player
        reward = 0.0

        # --- Decode and execute ------------------------------------------
        if action < self.num_cities:                       # MOVE
            self.pos[player] = action
            if self.pos[player] == self.pos[opp]:
                self._reveal_current_position(player)
            self.current_turn_actions[player].append(MOVE)
            self.moves_since_last_seen[opp] += 1

        elif action == self.WAIT_A:
            self.current_turn_actions[player].append(WAIT)
            self.moves_since_last_seen[opp] += 1

        elif action == self.CONTROL_A:
            self.city_states[self.pos[player]]["controlled_by"] = player
            self._reveal_current_position(player)
            self.current_turn_actions[player].append(CONTROL)

        elif action == self.DEEP_COVER_A:
            self.intel[player] -= INTEL_COST[DEEP_COVER]
            self.deep_cover[player] = True
            self.current_turn_actions[player].append(DEEP_COVER)

        elif action == self.PREP_A:
            self.intel[player] -= INTEL_COST[PREP]
            self.current_turn_actions[player].append(PREP)

        elif action == self.LOCATE_A:
            self.intel[player] -= INTEL_COST[LOCATE]
            self._reveal_current_position(opp)
            self.current_turn_actions[player].append(LOCATE)

        elif action == self.STRIKE_A:
            self.current_turn_actions[player].append(STRIKE)
            if self.pos[opp] == self.pos[player]:
                self.terminated = True
                reward = 1.0

        # --- End-of-action bookkeeping -----------------------------------
        self.actions_left -= 1
        if self.terminated:
            self._commit_turn(player)   # freeze the winning turn into history
            return self.observe(player), reward, True, False, {"action_mask": self._action_mask(player)}

        if self.actions_left == 0:
            self._end_turn(player)

        return (
            self.observe(self.current_player),
            reward,
            False,
            False,
            {"action_mask": self._action_mask(self.current_player)},
        )

    def render(self):
        print(f"[{self.map_name}] P1@{self.idx_to_city[self.pos[0]]} "
              f"P2@{self.idx_to_city[self.pos[1]]}  "
              f"turn=P{self.current_player + 1} acts_left={self.actions_left} "
              f"intel={self.intel} dc={self.deep_cover}")

    # --- Game helpers ------------------------------------------------------
    def _reveal_current_position(self, target: int) -> None:
        """Opponent of `target` learns `target`'s current position, unless deep cover is active."""
        if self.deep_cover[target]:
            return
        viewer = 1 - target
        self.last_seen[viewer] = self.pos[target]
        self.moves_since_last_seen[viewer] = 0

    def _commit_turn(self, player: int) -> None:
        """Flush the in-progress turn into action_history and record encryption state."""
        self.action_history[player].append(list(self.current_turn_actions[player]))
        self.encryption_active_at_turn[player].append(bool(self.unlocked[player][TECH_ENC]))
        self.current_turn_actions[player] = []

    def _end_turn(self, finishing: int) -> None:
        # Bank prep bonus for this player's next turn.
        self.pending_extra_actions[finishing] = sum(
            1 for a in self.current_turn_actions[finishing] if a == PREP
        )
        self._commit_turn(finishing)

        # Hand off.
        new_player = 1 - finishing
        self.current_player = new_player
        # Deep cover clears at the start of the covered player's own next turn.
        self.deep_cover[new_player] = False
        # Apply prep bonus accrued the last time this player acted.
        self.actions_left = 2 + self.pending_extra_actions[new_player]
        self.pending_extra_actions[new_player] = 0
        # Intel income: +1 per controlled small city, +4 per controlled large city.
        for i in range(self.num_cities):
            if self.city_states[i]["controlled_by"] == new_player:
                self.intel[new_player] += 4 if self.large_mask[i] else 1

    def _action_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.int8)
        my_city = self.pos[player]

        # MOVE: adjacent and not closed.
        for nb in self.neighbors[my_city]:
            if self.city_states[nb]["moves_till_closing"] > 0:
                mask[nb] = 1

        mask[self.WAIT_A] = 1
        if self.city_states[my_city]["controlled_by"] != player:
            mask[self.CONTROL_A] = 1
        if self.intel[player] >= INTEL_COST[DEEP_COVER]:
            mask[self.DEEP_COVER_A] = 1
        if self.intel[player] >= INTEL_COST[PREP]:
            mask[self.PREP_A] = 1
        if self.intel[player] >= INTEL_COST[LOCATE]:
            mask[self.LOCATE_A] = 1
        mask[self.STRIKE_A] = 1
        return mask

    # --- Observation -------------------------------------------------------
    def _format_history(self, owner: int, viewer: int) -> np.ndarray:
        """(K_TURNS, MAX_ACTIONS) view of `owner`'s history from `viewer`'s perspective.
        Row 0 = most recent completed turn. Unused slots = NOOP. Costed actions on
        encryption-active turns are HIDDEN when viewer != owner.
        """
        history = self.action_history[owner]
        enc_flags = self.encryption_active_at_turn[owner]
        out = np.full((K_TURNS, MAX_ACTIONS), NOOP, dtype=np.int32)
        for row_idx in range(K_TURNS):
            turn_idx = len(history) - 1 - row_idx
            if turn_idx < 0:
                break
            turn = history[turn_idx][-MAX_ACTIONS:]
            hide = enc_flags[turn_idx] and viewer != owner
            for col_idx, a in enumerate(turn):
                out[row_idx, col_idx] = HIDDEN if (hide and a in COSTED_ACTIONS) else a
        return out

    def observe(self, player: int) -> dict:
        assert player in (0, 1)
        opp = 1 - player
        n = self.num_cities

        moves_till_closing = np.fromiter(
            (self.city_states[i]["moves_till_closing"] for i in range(n)),
            dtype=np.int64, count=n,
        )
        controlled_by_me = np.fromiter(
            (self.city_states[i]["controlled_by"] == player for i in range(n)),
            dtype=np.int8, count=n,
        )
        controlled_by_opp = np.fromiter(
            (self.city_states[i]["controlled_by"] == opp for i in range(n)),
            dtype=np.int8, count=n,
        )
        amount_of_intel = np.fromiter(
            (self.city_states[i]["amount_of_intel"] for i in range(n)),
            dtype=np.int32, count=n,
        )

        return {
            "my_pos":                    self.pos[player],
            "last_seen_opponent":        self.last_seen[player],
            "opp_moves_since_last_seen": min(self.moves_since_last_seen[player], MAX_MOVES_SINCE_SEEN),
            "my_action_history":         self._format_history(player, player),
            "opp_action_history":        self._format_history(opp, player),
            "opp_unlocked_techs":        self.unlocked[opp].copy(),
            "my_intel":                  np.int32(self.intel[player]),
            "actions_left":              self.actions_left,
            "moves_till_closing":        moves_till_closing,
            "controlled_by_me":          controlled_by_me,
            "controlled_by_opp":         controlled_by_opp,
            "amount_of_intel":           amount_of_intel,
            "is_large":                  self.large_mask.copy(),
        }
