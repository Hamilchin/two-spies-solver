"""Interactive visual tester for TwoSpiesEnv.

- Click a city: attempt MOVE to it (ignored if illegal).
- Keys: w=WAIT  c=CONTROL  d=DEEP_COVER  p=PREP  l=LOCATE  s=STRIKE
        r=RESET  q=quit
Hot-seat: you play whoever `current_player` is; control hands off automatically.
"""

from __future__ import annotations
import sys

import matplotlib.pyplot as plt
import numpy as np

from env import (
    TwoSpiesEnv,
    CONTROL, DEEP_COVER, LOCATE, MOVE, PREP, STRIKE, NOOP,
    UNKNOWN_MOVE, UNKNOWN_INTEL_ACTION,
    TECH_RR, TECH_ENC, TECH_SR,
    NOT_CLOSING,
)

ACTION_NAMES = {
    MOVE: "MOVE", STRIKE: "STRIKE", CONTROL: "CONTROL",
    DEEP_COVER: "DEEP_COVER", PREP: "PREP", LOCATE: "LOCATE",
    NOOP: "NOOP",
}


def keyed_action(env: TwoSpiesEnv, key: str) -> int | None:
    """Return an action index for a non-move hotkey, or None if unbound.
    `w` is a shortcut for MOVE-to-current-city (i.e. wait)."""
    if key == "w":
        return env.pos[env.current_player]
    mapping = {
        "c": env.CONTROL_A, "d": env.DEEP_COVER_A,
        "p": env.PREP_A, "l": env.LOCATE_A, "s": env.STRIKE_A,
    }
    return mapping.get(key)


def nearest_city(env: TwoSpiesEnv, x: float, y: float, tol: float = 0.12) -> int | None:
    pos = env.layout()
    best_i, best_d = None, float("inf")
    for i, (cx, cy) in pos.items():
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    if best_i is None or best_d ** 0.5 > tol:
        return None
    return best_i


def draw_info(ax, env: TwoSpiesEnv, last_msg: str,
              total_rewards: list[float] | None = None) -> None:
    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cp = env.current_player

    def pos_str(p): return f"{env.idx_to_city[env.pos[p]]} ({env.pos[p]})"
    def seen_str(viewer):
        return f"{env.idx_to_city[env.last_seen[viewer]]} ({env.last_seen[viewer]})"
    def tech_str(p):
        u = env.unlocked[p]
        return f"RR:{int(u[TECH_RR])} ENC:{int(u[TECH_ENC])} SR:{int(u[TECH_SR])}"
    def bool_str(b): return "yes" if b else "—"
    def row(label, a, b):
        return f"  {label:<15}{str(a):<20}{b}"

    cur_actions = env.current_turn_actions[cp]
    cur_acts_str = " ".join(ACTION_NAMES.get(a, str(a)) for a in cur_actions) or "—"
    status = "TERMINATED" if env.terminated else "in progress"

    lines = [
        f"  P{cp+1} to act   actions_left: {env.actions_left}   [{status}]",
        "",
        f"  {'':<15}{'P1':<20}P2",
        f"  {'-'*14} {'-'*19} {'-'*14}",
        row("position",      pos_str(0),                      pos_str(1)),
        row("intel",         env.intel[0],                    env.intel[1]),
        row("deep cover",    bool_str(env.deep_cover[0]),     bool_str(env.deep_cover[1])),
        row("pending prep",  env.pending_extra_actions[0],    env.pending_extra_actions[1]),
        row("last saw opp",  seen_str(0),                     seen_str(1)),
        row("  moves ago",   env.moves_since_last_seen[0],    env.moves_since_last_seen[1]),
        row("techs",         tech_str(0),                     tech_str(1)),
    ]
    if total_rewards is not None:
        lines.append(row("ep reward",
                         f"{total_rewards[0]:+.2f}", f"{total_rewards[1]:+.2f}"))
    lines += [
        "",
        f"  Current turn (P{cp+1}):  {cur_acts_str}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=9)

    # Colored action-mask grid (green=legal, red=illegal).
    draw_action_mask(ax, env, start_y=0.42)

    # Footer.
    footer = [
        f"  Last event:  {last_msg}",
        "",
        "  Keys:  click=MOVE  w=WAIT  c=CONTROL  d=DEEP_COVER",
        "         p=PREP  l=LOCATE  s=STRIKE  r=RESET  q=quit",
    ]
    ax.text(0.02, 0.10, "\n".join(footer), ha="left", va="top",
            family="monospace", fontsize=8)

    ax.set_title(f"State  (P{cp+1} to act)")


def draw_action_mask(ax, env: TwoSpiesEnv, start_y: float = 0.42) -> None:
    """Render the action mask as a colored grid (green=legal, red=illegal)."""
    mask = env.action_mask(env.current_player)
    GREEN = "#2a8a2a"
    RED   = "#a83222"

    ax.text(0.02, start_y, "Action mask  (green=legal, red=illegal):",
            fontsize=8, family="monospace", ha="left", va="top", weight="bold")

    n = env.num_cities
    rows_per_col = (n + 1) // 2
    line_h = 0.026

    # MOVE actions in two columns.
    for i in range(n):
        col = i // rows_per_col
        rix = i % rows_per_col
        x = 0.02 + col * 0.48
        y = start_y - 0.03 - line_h * rix
        color = GREEN if mask[i] else RED
        ax.text(x, y, f"  M→{i:>2}: {env.idx_to_city[i][:11]}",
                color=color, fontsize=8, family="monospace",
                ha="left", va="top")

    # Special actions in one row beneath the MOVE grid.
    specials = [
        ("CONTROL",   env.CONTROL_A),
        ("DEEP_CVR",  env.DEEP_COVER_A),
        ("PREP",      env.PREP_A),
        ("LOCATE",    env.LOCATE_A),
        ("STRIKE",    env.STRIKE_A),
    ]
    spec_y = start_y - 0.03 - line_h * rows_per_col - 0.01
    for i, (name, a) in enumerate(specials):
        color = GREEN if mask[a] else RED
        ax.text(0.02 + i * 0.19, spec_y, name,
                color=color, fontsize=8, family="monospace",
                ha="left", va="top", weight="bold")


def draw_pov(ax, env: TwoSpiesEnv, obs: dict, pov_player: int) -> None:
    """Side-panel rendering of `pov_player`'s observation at the current frame."""
    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def fmt_cell(a):
        a = int(a)
        if a == NOOP:                 return "·"
        if a == UNKNOWN_MOVE:         return "?mv"
        if a == UNKNOWN_INTEL_ACTION: return "?int"
        return ACTION_NAMES.get(a, str(a))[:5]

    def fmt_history(mat):
        lines = []
        for i, row in enumerate(mat):
            cells = " ".join(f"{fmt_cell(a):<5}" for a in row)
            age = "now" if i == 0 else f"t-{i}"
            lines.append(f"    {age:<4} | {cells}")
        return lines

    pu = obs["opp_unlocked_techs"]
    city_rows = [f"    {'idx':<3} {'city':<11} {'sz':<2} {'own':<4} {'state':<7} intel"]
    for i in range(env.num_cities):
        owner = ("me"  if obs["controlled_by_me"][i]  else
                 "opp" if obs["controlled_by_opp"][i] else "—")
        mtc = int(obs["moves_till_closing"][i])
        state = "open" if mtc == NOT_CLOSING else "closed" if mtc == 0 else f"×{mtc}"
        sz = "L" if obs["is_large"][i] else "s"
        intel_here = int(obs["amount_of_intel"][i])
        name = env.idx_to_city[i][:11]
        city_rows.append(f"    {i:<3} {name:<11} {sz:<2} {owner:<4} {state:<7} {intel_here}")

    lines = [
        f"  P{pov_player+1} POV",
        "",
        f"  my_pos:           {env.idx_to_city[obs['my_pos']]} ({obs['my_pos']})",
        f"  my_intel:         {int(obs['my_intel'])}",
        f"  actions_left:     {obs['actions_left']}",
        f"  last_seen_opp:    {env.idx_to_city[obs['last_seen_opponent']]} ({obs['last_seen_opponent']})",
        f"  moves_since_seen: {obs['opp_moves_since_last_seen']}",
        f"  opp_techs:        RR:{int(pu[TECH_RR])} ENC:{int(pu[TECH_ENC])} SR:{int(pu[TECH_SR])}",
        "",
        "  my action history:",
        *fmt_history(obs["my_action_history"]),
        "",
        "  opp action history (as seen by me):",
        *fmt_history(obs["opp_action_history"]),
        "",
        "  cities:",
        *city_rows,
    ]
    ax.text(0.01, 0.99, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=8)


def replay(snapshots: list[dict], map_or_path) -> None:
    """Slider-driven playback of a snapshot log. A radio button toggles
    between ground-truth info and the current player's POV (which also
    applies fog-of-war to the opponent on the graph)."""
    from matplotlib.widgets import Slider, RadioButtons

    env = TwoSpiesEnv(map_or_path)
    env.reset()
    state = {"frame": 0, "mode": "truth"}

    fig = plt.figure(figsize=(14, 8))
    ax_graph  = fig.add_axes([0.02, 0.18, 0.54, 0.78])
    ax_info   = fig.add_axes([0.60, 0.18, 0.38, 0.78])
    ax_slider = fig.add_axes([0.18, 0.06, 0.70, 0.03])
    ax_radio  = fig.add_axes([0.02, 0.02, 0.13, 0.09])

    n_last = max(len(snapshots) - 1, 0)
    slider = Slider(ax_slider, "frame", 0, max(n_last, 1), valinit=0, valstep=1)
    radio  = RadioButtons(ax_radio, ("truth", "POV"))

    def redraw():
        i = min(state["frame"], n_last)
        snap = snapshots[i]
        env.load_snapshot(snap)
        ax_graph.clear()
        pov = snap["current_player"] if state["mode"] == "POV" else None
        env.draw(ax_graph, pov=pov)
        if state["mode"] == "POV":
            draw_pov(ax_info, env, snap["obs"], snap["current_player"])
        else:
            draw_info(ax_info, env, f"frame {i}/{n_last}")
        fig.canvas.draw_idle()

    def on_slider(val):
        state["frame"] = int(val)
        redraw()

    def on_radio(label):
        state["mode"] = label
        redraw()

    slider.on_changed(on_slider)
    radio.on_clicked(on_radio)

    redraw()
    plt.show()


def main() -> None:
    map_path = sys.argv[1] if len(sys.argv) > 1 else "maps/roundabout.json"
    env = TwoSpiesEnv(map_path)
    obs, info = env.reset()
    last_msg = {"text": "reset"}
    total_rewards = [0.0, 0.0]   # accumulated per-player reward for this episode

    fig, (ax_graph, ax_info) = plt.subplots(1, 2, figsize=(13, 7),
                                            gridspec_kw={"width_ratios": [3, 2]})

    def repaint():
        ax_graph.clear()
        env.draw(ax_graph)
        draw_info(ax_info, env, last_msg["text"], total_rewards=total_rewards)
        fig.canvas.draw_idle()

    def try_step(action: int, label: str) -> None:
        mask = env.action_mask(env.current_player)
        if not mask[action]:
            last_msg["text"] = f"ILLEGAL: {label} (mask={int(mask[action])})"
            return
        actor = env.current_player
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards[actor] += float(reward)
        if terminated:                          # zero-sum: loser gets -1.0
            total_rewards[1 - actor] -= 1.0
        last_msg["text"] = (
            f"{label}  reward={reward:+.2f}  terminated={terminated}"
        )

    def on_click(event):
        if event.inaxes is not ax_graph or event.xdata is None:
            return
        i = nearest_city(env, event.xdata, event.ydata)
        if i is None:
            last_msg["text"] = "click: no city near cursor"
            repaint()
            return
        try_step(i, f"MOVE -> {i}:{env.idx_to_city[i]}")
        repaint()

    def on_key(event):
        k = (event.key or "").lower()
        if k == "q":
            plt.close(fig); return
        if k == "r":
            env.reset()
            total_rewards[:] = [0.0, 0.0]
            last_msg["text"] = "reset"
            repaint(); return
        a = keyed_action(env, k)
        if a is None:
            return
        labels = {
            env.CONTROL_A: "CONTROL", env.DEEP_COVER_A: "DEEP_COVER",
            env.PREP_A: "PREP", env.LOCATE_A: "LOCATE",
            env.STRIKE_A: "STRIKE",
        }
        if a in labels:
            label = labels[a]
        else:
            label = f"MOVE -> {a}:{env.idx_to_city[a]} (wait)"
        try_step(a, label)
        repaint()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    repaint()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
