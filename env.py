from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- Action IDs (used in action_history) ------------------------------------
# Waiting is encoded as MOVE to the current city (self-move), so there is no
# separate WAIT action id.
MOVE, STRIKE, CONTROL, DEEP_COVER, PREP, LOCATE, UNLOCK_RR, UNLOCK_ENC, UNLOCK_SR = range(9)
NOOP = 9
UNKNOWN_INTEL_ACTION = 10    # costed action with encryption active, from opp POV
UNKNOWN_MOVE = 11            # any opp MOVE, from opp POV (we never know the target)
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
MAX_MOVES_SINCE_SEEN = 6
# moves_till_closing: 0 = closed, 1 or 2 = closing soon, 3 = not yet marked closing.
NOT_CLOSING = 3
CLOSING_STATES = 4  # values in {0, 1, 2, 3}


class TwoSpiesEnv(gym.Env):
    """Two Spies Gym env.

    Action space layout: Discrete(num_cities + 5), indexed as
        0 .. N-1 :  MOVE to city i (self-move to current city = WAIT)
        N        :  CONTROL
        N+1      :  DEEP_COVER
        N+2      :  PREP
        N+3      :  LOCATE
        N+4      :  STRIKE
    `info["action_mask"]` flags which indices are currently legal.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, map_or_path, *,
                 max_turns: int = 100,
                 intel_reward: float = 0.005,
                 world_events: bool = False,
                 closing_start_turn: int = 10):
        super().__init__()
        self.max_turns: int = max_turns
        self.intel_reward: float = intel_reward
        self.world_events: bool = world_events
        self.closing_start_turn: int = closing_start_turn

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
        self.CONTROL_A:    int = n
        self.DEEP_COVER_A: int = n + 1
        self.PREP_A:       int = n + 2
        self.LOCATE_A:     int = n + 3
        self.STRIKE_A:     int = n + 4
        self.num_actions:  int = n + 5

        # Episode state (populated by reset).
        self.pos: list[int] = [0, 0]
        self.last_seen: list[int] = [0, 0]                # last_seen[viewer] = where viewer last saw the opponent
        self.moves_since_last_seen: list[int] = [0, 0]
        self.intel: list[int] = [0, 0]
        self.unlocked: list[np.ndarray] = [
            np.zeros(NUM_TECHS, dtype=np.int8),
            np.zeros(NUM_TECHS, dtype=np.int8),
        ]
        self.deep_cover: list[bool] = [False, False]
        self.pending_extra_actions: list[int] = [0, 0]
        self.current_turn_actions: list[list[int]] = [[], []]
        # Completed turns: each entry is a list of action IDs (any length).
        self.action_history: list[list[list[int]]] = [[], []]
        self.encryption_active_at_turn: list[list[bool]] = [[], []]
        self.current_player: int = 0
        self.actions_left: int = 2
        self.terminated: bool = False
        self.current_turn: int = 1
        self.intel_gained_this_turn: list[int] = [0, 0]

        self.city_states: dict[int, dict] = {}

        # Cached graph layout for rendering (computed lazily on first draw).
        self._layout: dict[int, tuple[float, float]] | None = None

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
        self.current_turn = 1
        self.intel_gained_this_turn = [0, 0]

        self.city_states = {
            i: {
                "moves_till_closing": NOT_CLOSING,
                "controlled_by":      None,   # None | 0 | 1
                "amount_of_intel":    0,
                "is_large":           bool(self.large_mask[i]),
                "last_intel_turn":    None,   # last turn intel was spawned here (world_events)
            }
            for i in range(self.num_cities)
        }

        # Fire turn-1's world events so intel can already be on the board when
        # P1 takes their first action. Closing is gated by closing_start_turn,
        # so this only spawns intel at the very start.
        self._world_step()

        return self.observe(self.current_player), {"action_mask": self.action_mask(self.current_player)}

    def step(self, action: int):
        if self.terminated:
            raise RuntimeError("step() called on terminated episode")
        if not self.action_mask(self.current_player)[action]:
            raise ValueError(f"illegal action {action} for player {self.current_player}")

        player = self.current_player
        opp = 1 - player
        reward = 0.0

        # --- Decode and execute ------------------------------------------
        if action < self.num_cities:                       # MOVE (self-move = wait)
            self.pos[player] = action
            self.current_turn_actions[player].append(MOVE)
            # Tick the opponent's fog clock, THEN reveal if we stepped into a
            # city the opponent controls — reveal has the last word and resets
            # the counter to 0 when applicable.
            self.moves_since_last_seen[opp] += 1
            if self.city_states[self.pos[player]]["controlled_by"] == opp:
                self._reveal_current_position(player)

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
            return self.observe(player), reward, True, False, {"action_mask": self.action_mask(player)}

        truncated = False
        if self.actions_left == 0:
            # Dense reward for intel collected during the turn we just closed.
            reward += self.intel_reward * self.intel_gained_this_turn[player]
            self._end_turn(player)
            if self.current_turn > self.max_turns:
                truncated = True

        return (
            self.observe(self.current_player),
            reward,
            False,
            truncated,
            {"action_mask": self.action_mask(self.current_player)},
        )

    def render(self):
        print(f"[{self.map_name}] P1@{self.idx_to_city[self.pos[0]]} "
              f"P2@{self.idx_to_city[self.pos[1]]}  "
              f"turn=P{self.current_player + 1} acts_left={self.actions_left} "
              f"intel={self.intel} dc={self.deep_cover}")

    # --- Drawing -----------------------------------------------------------
    def layout(self) -> dict[int, tuple[float, float]]:
        """Return (and cache) a 2D layout mapping city index -> (x, y)."""
        if self._layout is None:
            import networkx as nx
            g = nx.Graph()
            g.add_nodes_from(range(self.num_cities))
            for i, nbrs in enumerate(self.neighbors):
                for j in nbrs:
                    if i < j:
                        g.add_edge(i, j)
            self._layout = nx.spring_layout(g, seed=0)
        return self._layout

    def draw(self, ax, pov: int | None = None) -> None:
        """Paint the current env state onto a matplotlib Axes.
        If `pov` is a player index, apply fog-of-war: the opponent (relative
        to `pov`) floats above their last-seen city with reduced opacity
        whenever their true position isn't currently known.
        """
        pos = self.layout()
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{self.map_name}")

        # Edges: solid if both endpoints open, faint gray if either is closed.
        for i in range(self.num_cities):
            for j in self.neighbors[i]:
                if j <= i:
                    continue
                i_closed = self.city_states[i]["moves_till_closing"] == 0
                j_closed = self.city_states[j]["moves_till_closing"] == 0
                color = "#dddddd" if (i_closed or j_closed) else "#888888"
                (xi, yi), (xj, yj) = pos[i], pos[j]
                ax.plot([xi, xj], [yi, yj], color=color, zorder=1,
                        linewidth=1.0)

        # Nodes.
        for i in range(self.num_cities):
            x, y = pos[i]
            cs = self.city_states[i]
            mtc = cs["moves_till_closing"]
            closed = mtc == 0
            closing = 0 < mtc < NOT_CLOSING
            size = 500 if cs["is_large"] else 280
            # Fill color by controller; closed cities are blacked out.
            owner = cs["controlled_by"]
            if closed:
                face = "#111111"
            elif owner == 0:
                face = "#a9cbe8"
            elif owner == 1:
                face = "#eab3a9"
            else:
                face = "#ffffff"
            edge = "#222222"
            style = "dashed" if closing else "solid"
            ax.scatter([x], [y], s=size, facecolor=face, edgecolor=edge,
                       linewidths=1.6, linestyle=style, zorder=2)

            # Warning indicator: small black dot for 2-turn warning, larger
            # black dot for 1-turn warning. Offset to the upper-right so it
            # doesn't obscure the intel number.
            if closing:
                dot_size = 80 if mtc == 1 else 28
                ax.scatter([x + 0.04], [y + 0.04], s=dot_size,
                           facecolor="black", edgecolor="white",
                           linewidths=1.0, zorder=4)

            # Intel count inside the node.
            if cs["amount_of_intel"] > 0:
                text_color = "white" if closed else "#333"
                ax.text(x, y, str(cs["amount_of_intel"]),
                        ha="center", va="center", fontsize=9,
                        color=text_color, zorder=3)

            # Label below: "i: Name" (with moves_till_closing if closing soon).
            label = f"{i}: {self.idx_to_city[i]}"
            if closing:
                label += f"  (×{cs['moves_till_closing']})"
            elif closed:
                label += "  (closed)"
            ax.text(x, y - 0.09, label, ha="center", va="top",
                    fontsize=8, zorder=3)

        # Spies: solid dot for current player, hollow for the other.
        # In POV mode, the non-`pov` spy floats above their last-seen city
        # at low alpha whenever their true position isn't currently known.
        for p, color in [(0, "#1f4e8a"), (1, "#a83222")]:
            fog = pov is not None and p != pov and self.moves_since_last_seen[pov] > 0
            city = self.last_seen[pov] if fog else self.pos[p]
            x, y = pos[city]
            dx = -0.025 if p == 0 else 0.025
            dy = 0.08 if fog else 0.03
            alpha = 0.4 if fog else 1.0
            if p == self.current_player:
                ax.scatter([x + dx], [y + dy], s=140, facecolor=color,
                           edgecolor="black", linewidths=1.2, zorder=4, alpha=alpha)
            else:
                ax.scatter([x + dx], [y + dy], s=140, facecolor="white",
                           edgecolor=color, linewidths=2.0, zorder=4, alpha=alpha)
            ax.text(x + dx, y + dy, f"P{p+1}", ha="center", va="center",
                    fontsize=7, color="black", zorder=5, alpha=alpha)

    # --- Snapshots (Pattern-B replay playback) ----------------------------
    def snapshot(self) -> dict:
        """Serialize state plus the current player's observation.
        Pairs with load_snapshot() — restoring a snapshot reproduces
        everything draw() and observe() read."""
        return {
            "pos": list(self.pos),
            "last_seen": list(self.last_seen),
            "moves_since_last_seen": list(self.moves_since_last_seen),
            "intel": list(self.intel),
            "unlocked": [u.copy() for u in self.unlocked],
            "deep_cover": list(self.deep_cover),
            "pending_extra_actions": list(self.pending_extra_actions),
            "current_turn_actions": [list(a) for a in self.current_turn_actions],
            "action_history": [[list(t) for t in h] for h in self.action_history],
            "encryption_active_at_turn": [list(e) for e in self.encryption_active_at_turn],
            "current_player": self.current_player,
            "actions_left": self.actions_left,
            "terminated": self.terminated,
            "current_turn": self.current_turn,
            "intel_gained_this_turn": list(self.intel_gained_this_turn),
            "city_states": {i: dict(cs) for i, cs in self.city_states.items()},
            "obs": self.observe(self.current_player),
        }

    def load_snapshot(self, snap: dict) -> None:
        """Restore state from a snapshot dict (inverse of snapshot())."""
        self.pos = list(snap["pos"])
        self.last_seen = list(snap["last_seen"])
        self.moves_since_last_seen = list(snap["moves_since_last_seen"])
        self.intel = list(snap["intel"])
        self.unlocked = [u.copy() for u in snap["unlocked"]]
        self.deep_cover = list(snap["deep_cover"])
        self.pending_extra_actions = list(snap["pending_extra_actions"])
        self.current_turn_actions = [list(a) for a in snap["current_turn_actions"]]
        self.action_history = [[list(t) for t in h] for h in snap["action_history"]]
        self.encryption_active_at_turn = [list(e) for e in snap["encryption_active_at_turn"]]
        self.current_player = snap["current_player"]
        self.actions_left = snap["actions_left"]
        self.terminated = snap["terminated"]
        self.current_turn = snap["current_turn"]
        self.intel_gained_this_turn = list(snap["intel_gained_this_turn"])
        self.city_states = {i: dict(cs) for i, cs in snap["city_states"].items()}

    # --- External game-master events --------------------------------------
    # These are driven from outside the env (real-game mirror, scripted
    # scenarios, random event generators during training).
    def spawn_intel(self, city: int, amount: int = 1) -> None:
        """Add `amount` pickupable intel to `city`."""
        self.city_states[city]["amount_of_intel"] += amount

    def begin_closing(self, city: int, turns: int = 2) -> None:
        """Start `city` closing: `turns` until closed (0 = closed now)."""
        self.city_states[city]["moves_till_closing"] = turns

    # --- World step (auto-driven events when world_events=True) -----------
    # Replicates the actual Two Spies city-closing and intel-spawning logic.
    def _world_step(self) -> None:
        """Tick warnings, maybe begin closing a new city, maybe spawn intel."""
        if not self.world_events:
            return

        # 1. Decrement any city already in the warning phase: 2 -> 1 -> 0 (closed).
        for cs in self.city_states.values():
            if 0 < cs["moves_till_closing"] < NOT_CLOSING:
                cs["moves_till_closing"] -= 1

        # 2. If no city is currently in warning, pick a new one to begin closing
        #    (gated by closing_start_turn so the early game stays open).
        if self.current_turn >= self.closing_start_turn:
            in_warning = any(
                0 < cs["moves_till_closing"] < NOT_CLOSING
                for cs in self.city_states.values()
            )
            if not in_warning:
                chosen = self._find_next_city_to_close()
                if chosen is not None:
                    self.city_states[chosen]["moves_till_closing"] = 2

        # 3. Maybe spawn intel somewhere.
        self._spawn_random_intel()

    def _find_next_city_to_close(self) -> int | None:
        """Pick a city to begin closing. Mirrors the friend's findNextCityToClose
        heuristic: skip articulation points, prefer cities far from the graph
        'center' with few connections, avoid leaving neighbors isolated, halve
        chance for large cities. Returns a city index or None if nothing eligible."""
        import networkx as nx

        # Passable subgraph = anything not closed (open + warning).
        passable = {i for i in range(self.num_cities)
                    if self.city_states[i]["moves_till_closing"] > 0}
        open_cities = [i for i in range(self.num_cities)
                       if self.city_states[i]["moves_till_closing"] == NOT_CLOSING]

        if len(open_cities) <= 1:
            return None

        # Build the passable subgraph once; we reuse it for both articulation
        # detection and graph-distance computations.
        g = nx.Graph()
        g.add_nodes_from(passable)
        for i in passable:
            for j in self.neighbors[i]:
                if j in passable and j > i:
                    g.add_edge(i, j)
        articulation = set(nx.articulation_points(g))

        best_city, best_chance = None, float("-inf")
        weighted: list[tuple[int, int]] = []

        for city in open_cities:
            if city in articulation:
                continue
            chance = self._close_chance(city, g, passable)
            if chance > best_chance:
                best_city, best_chance = city, chance
            if chance > 0:
                weighted.append((city, chance))

        if not weighted:
            return best_city
        return self._weighted_choice(weighted)

    def _close_chance(self, city: int, g, passable: set[int]) -> int:
        """Score for choosing this city to close. Higher = more likely.
        Uses graph distance (not screen coordinates) so the heuristic is
        layout-invariant."""
        import networkx as nx

        # 'Peripheral-ness' = average shortest-path distance from this city to
        # the rest of the passable subgraph. Bigger = more peripheral.
        dists = nx.single_source_shortest_path_length(g, city)
        others = [d for k, d in dists.items() if k != city]
        avg_dist = sum(others) / len(others) if others else 0.0
        chance = int(avg_dist * 5)   # tuning: peripherals get +5 per hop of avg distance

        open_nbrs = [j for j in self.neighbors[city] if j in passable]
        if not open_nbrs:
            return 0

        # Bonus for low connectivity.
        chance += self._connection_bonus(len(open_nbrs))

        # Penalty for leaving neighbors isolated.
        for nbr in open_nbrs:
            iso = sum(1 for k in self.neighbors[nbr]
                      if k != city and k in passable)
            chance -= self._isolation_penalty(iso)

        if bool(self.large_mask[city]):
            chance = chance // 2
        return chance

    @staticmethod
    def _connection_bonus(n: int) -> int:
        # Few-connection cities get a small bonus toward closing.
        return {1: 5, 2: 3, 3: 1}.get(n, 0)

    @staticmethod
    def _isolation_penalty(iso: int) -> int:
        # Large penalty if a neighbor would be left totally isolated.
        return {0: 20, 1: 5}.get(iso, 0)

    def _weighted_choice(self, options: list[tuple[int, int]]) -> int:
        """Pick one (item, weight) by weight using the env's seeded RNG."""
        weights = np.array([w for _, w in options], dtype=np.float64)
        probs = weights / weights.sum()
        idx = int(self.np_random.choice(len(options), p=probs))
        return options[idx][0]

    def _spawn_random_intel(self) -> None:
        """Maybe pick a city and spawn intel there, per the friend's intel logic."""
        intel_exists = any(
            cs["amount_of_intel"] > 0 for cs in self.city_states.values()
        )
        # 50% per turn while intel exists; otherwise always.
        if intel_exists and int(self.np_random.integers(0, 100)) >= 50:
            return

        weighted: list[tuple[int, int]] = []
        for i in range(self.num_cities):
            chance = self._intel_chance(i)
            if chance > 0:
                weighted.append((i, chance))
        if not weighted:
            return

        chosen = self._weighted_choice(weighted)
        amount = int(self.np_random.integers(15, 26))   # 15..25 intel per drop
        self.city_states[chosen]["amount_of_intel"] += amount
        self.city_states[chosen]["last_intel_turn"] = self.current_turn

    def _intel_chance(self, city: int) -> int:
        """Score for spawning intel here. Higher = more likely chosen."""
        cs = self.city_states[city]

        # Hard vetoes
        last = cs["last_intel_turn"]
        if last is not None and last > self.current_turn - 2:
            return 0
        if cs["moves_till_closing"] != NOT_CLOSING:   # any closing/closed city
            return 0
        if cs["amount_of_intel"] >= 50:
            return 0
        if self.pos[0] == city or self.pos[1] == city:   # don't drop on a player
            return 0

        chance = 5  # baseline
        if 0 < cs["amount_of_intel"] < 30:
            chance += 75
        if bool(self.large_mask[city]):
            chance += 40
        if cs["controlled_by"] is None:
            chance += 5
            chance *= 3
        return chance

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
        # Hand off and run start-of-turn logic for the new current player.
        self.current_player = 1 - finishing
        self.current_turn += 1
        self._start_turn(self.current_player)

    def _start_turn(self, player: int) -> None:
        """Run start-of-turn bookkeeping for `player`."""
        # World events (auto-driven city closures + intel spawns) fire first
        # so the player can perceive the updated map this turn.
        self._world_step()
        # Deep cover expires at the start of the covered player's own next turn.
        self.deep_cover[player] = False
        # Apply prep bonus accrued the last time this player acted.
        self.actions_left = 2 + self.pending_extra_actions[player]
        self.pending_extra_actions[player] = 0
        # Tally intel collected this turn (income + pickup) for the dense reward.
        gained = 0
        for i in range(self.num_cities):
            if self.city_states[i]["controlled_by"] == player:
                gained += 4 if self.large_mask[i] else 1
        here = self.city_states[self.pos[player]]
        gained += here["amount_of_intel"]
        here["amount_of_intel"] = 0
        self.intel[player] += gained
        self.intel_gained_this_turn[player] = gained
        # Start-of-turn reveals (deep cover just expired above for `player`):
        #   * if you start the turn in an opponent-controlled city, your cover
        #     is blown,
        #   * if you share a city with the opponent, they are revealed to you
        #     (unless their deep cover is active).
        opp = 1 - player
        if self.city_states[self.pos[player]]["controlled_by"] == opp:
            self._reveal_current_position(player)
        if self.pos[player] == self.pos[opp]:
            self._reveal_current_position(opp)

    def action_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.int8)
        my_city = self.pos[player]
        my_cs = self.city_states[my_city]
        my_mtc = my_cs["moves_till_closing"]
        here_closing = 0 < my_mtc < NOT_CLOSING
        here_closed  = my_mtc == 0

        # MOVE: any neighbor that isn't closed.
        for nb in self.neighbors[my_city]:
            if self.city_states[nb]["moves_till_closing"] > 0:
                mask[nb] = 1

        # Forced evacuation: only MOVE-out is legal if we're already at a
        # closed city, or if we're in a closing city on our last action.
        # (The world step is expected to keep at least one open neighbor reachable.)
        if here_closed or (here_closing and self.actions_left == 1):
            return mask

        # Self-move (WAIT) is always legal otherwise.
        mask[my_city] = 1

        if my_cs["controlled_by"] != player:
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
        Row 0 = most recent completed turn. Unused slots = NOOP.
        From an opponent's POV:
          * MOVE is always replaced by UNKNOWN_MOVE (target city never leaks),
          * costed actions on encryption-active turns become UNKNOWN_INTEL_ACTION.
        """
        history = self.action_history[owner]
        enc_flags = self.encryption_active_at_turn[owner]
        out = np.full((K_TURNS, MAX_ACTIONS), NOOP, dtype=np.int32)
        foreign = viewer != owner
        for row_idx in range(K_TURNS):
            turn_idx = len(history) - 1 - row_idx
            if turn_idx < 0:
                break
            turn = history[turn_idx][-MAX_ACTIONS:]
            hide_intel = foreign and enc_flags[turn_idx]
            for col_idx, a in enumerate(turn):
                if foreign and a == MOVE:
                    out[row_idx, col_idx] = UNKNOWN_MOVE
                elif hide_intel and a in COSTED_ACTIONS:
                    out[row_idx, col_idx] = UNKNOWN_INTEL_ACTION
                else:
                    out[row_idx, col_idx] = a
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
