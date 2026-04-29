#!/usr/bin/env python
"""Local Streamlit app for per-step PPO bin/action inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.ppo_debug import (
    ADD_ACTION_FEATURE_LABELS,
    GLOBAL_FEATURE_NAMES,
    PPODebugSession,
    STOP_ACTION_FEATURE_LABELS,
    build_gt_boundary_bin_centers,
    build_gt_outline_segments,
)


COLOR_OPTIONS: tuple[str, ...] = (
    "reward_total",
    "policy_probability",
    "expr_weighted",
    "expr_old_weighted",
    "distance_penalty",
    "overlap_penalty",
    "neighbor_bonus",
    "candidate_to_current_centroid_distance",
    "candidate_compactness_gain",
)

DEBUG_SESSION_SCHEMA_VERSION = "2026-04-29-w5-expression-v1"

REPLAY_METRIC_HELP: dict[str, str] = {
    "Replay Steps": "Number of decision steps replayed for this cell.",
    "Replay Assigned": "Number of bins assigned at the end of replay.",
    "Replay Reward": "Sum of all step rewards along the replayed trajectory.",
    "Replay Source": "saved_trace means exact replay from saved evaluate step traces. greedy_policy means fallback re-run by the current greedy policy.",
    "Replay Match": "Whether replayed total reward, step count, and final assigned bin count match the eval row for this cell.",
}

BIN_EXPRESSION_HELP = (
    "Select any candidate bin for this cell and inspect its selected-gene expression vector. "
    "Top genes are sorted by raw bin-level expression among the genes used by the PPO reference matrix."
)

SECTION_HELP: dict[str, str] = {
    "State Summary": "State-level features describing current size, growth, remaining frontier quality, and critic value.",
    "Reward Decomposition": "Immediate add-reward breakdown for the current greedy add bin at this step.",
    "Policy / Shape Diagnostics": "Actor preference and dynamic shape features for the current greedy add bin at this step.",
    "STOP Panel": "STOP action score, probability, and remaining-frontier stop statistic.",
}

STATE_METRIC_HELP: dict[str, str] = {
    "assigned_frac": "Current assigned bins divided by all candidate bins in this episode.",
    "step_frac": "Current step index divided by the episode max-step budget.",
    "remaining_frac": "Fraction of candidate bins not yet assigned.",
    "grow_ratio_scaled": "Current assigned_count / seed_count, log-scaled into [0, 1].",
    "positive_frontier_fraction": "Fraction of frontier bins whose add reward is positive.",
    "centroid_drift_scaled": "Distance from current assigned centroid to nucleus center, scaled by r_max_um.",
    "compactness_proxy": "Average 8-neighbor support among assigned bins. Higher means more connected.",
    "frontier_add_reward_topk_mean": "Mean of the current top-k frontier add rewards used by STOP.",
    "frontier_add_reward_mean": "Mean add reward over current frontier bins.",
    "frontier_add_reward_std": "Std of add reward over current frontier bins.",
    "frontier_add_reward_max": "Best current frontier add reward.",
    "seed_compactness": "Average 8-neighbor support inside the initial nuclear seed.",
    "seed_radius_p90_scaled": "90th percentile seed-bin radius from nucleus center, scaled by r_max_um.",
    "seed_aspect_ratio_scaled": "Initial seed elongation proxy; 0 is rounder, 1 is more elongated.",
    "value_estimate": "Critic estimate of expected future return from the current state.",
}

STOP_METRIC_HELP: dict[str, str] = {
    "stop_prob": "Policy probability of choosing STOP now.",
    "stop_reward": "Immediate STOP reward at this step.",
    "stop_logit": "Raw actor score for STOP before softmax.",
    "stop_delta": "Current stop statistic computed from remaining frontier add rewards.",
}

REWARD_DECOMP_HELP: dict[str, str] = {
    "reward_total": "Full immediate add reward for this candidate bin.",
    "expr_raw": "Posterior-confidence gain from adding this bin before z-scoring and weighting.",
    "expr_conf": "Confidence multiplier from total counts, c / (c + a).",
    "expr_term": "Posterior-confidence gain term after optional frontier z-scoring.",
    "w1_expr": "Expression contribution after multiplying by w1.",
    "expr_old_raw": "Old bin-posterior compatibility score before z-scoring and weighting.",
    "expr_old_term": "Old compatibility term after optional frontier z-scoring.",
    "w5_expr_old": "Old compatibility helper contribution after multiplying by w5.",
    "w2_p_dis": "Distance penalty contribution from w2 * p_dis.",
    "w3_p_overlap": "Overlap penalty contribution from w3 * p_overlap.",
    "w4_neighbor": "Neighbor-support bonus contribution from w4 * neighbor_support.",
    "neighbor_support": "Fraction of the 8 neighbors already assigned.",
}

POLICY_SHAPE_HELP: dict[str, str] = {
    "policy_prob": "Policy probability for this add action at the current step.",
    "policy_logit": "Raw actor score for this add action before softmax.",
    "centroid_distance": "Candidate distance to current assigned centroid, scaled by r_max_um.",
    "compactness_gain": "Estimated compactness change if this bin were added next.",
    "expr_frontier_mean": "Mean raw expression score over current frontier bins.",
    "expr_frontier_std": "Std of raw expression score over current frontier bins.",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ppo-eval-run-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cell-id", type=str, default=None)
    return parser.parse_known_args(sys.argv[1:])[0]


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stPopover"] button {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            outline: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
            height: auto !important;
            line-height: 1 !important;
        }
        div[data-testid="stPopover"] button:hover,
        div[data-testid="stPopover"] button:focus,
        div[data-testid="stPopover"] button:active {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }
        div[data-testid="stPopover"] button p {
            margin: 0 !important;
            color: #6b7280 !important;
            font-size: 0.95rem !important;
            line-height: 1 !important;
        }
        div[data-testid="stMetric"] {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_session(
    ppo_eval_run_dir: str,
    checkpoint: str | None,
    device: str,
    session_schema_version: str,
) -> PPODebugSession:
    return PPODebugSession.from_eval_run(
        ppo_eval_run_dir=ppo_eval_run_dir,
        checkpoint_path=checkpoint,
        device_name=device,
    )


def _series_color_kwargs(metric_name: str, values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {
            "colorscale": "Viridis",
        }
    if metric_name in {"reward_total", "expr_weighted", "expr_old_weighted", "candidate_compactness_gain"}:
        max_abs = float(max(np.max(np.abs(finite)), 1.0e-6))
        return {
            "colorscale": "RdBu",
            "cmin": -max_abs,
            "cmax": max_abs,
            "cmid": 0.0,
        }
    return {
        "colorscale": "Viridis",
        "cmin": float(np.min(finite)),
        "cmax": float(np.max(finite)) if finite.size > 0 else 1.0,
    }


def _build_metric_hover(df: pd.DataFrame, color_by: str) -> list[str]:
    out: list[str] = []
    for row in df.itertuples(index=False):
        out.append(
            "<br>".join(
                (
                    f"bin_idx={int(row.bin_idx)}",
                    f"barcode={row.barcode}",
                    f"is_frontier={bool(row.is_frontier)}",
                    f"is_assigned={bool(row.is_assigned)}",
                    f"reward={float(row.reward_total):.4f}",
                    f"prob={float(row.policy_probability):.4f}",
                    f"{color_by}={float(getattr(row, color_by)):.4f}",
                )
            )
        )
    return out


def _add_bin_trace(
    fig: go.Figure,
    *,
    df: pd.DataFrame,
    name: str,
    marker_kwargs: dict[str, Any],
    color_by: str,
    legendgroup: str | None = None,
) -> None:
    if df.empty:
        return
    fig.add_trace(
        go.Scattergl(
            x=df["x_um"],
            y=df["y_um"],
            mode="markers",
            name=name,
            legendgroup=legendgroup or name,
            customdata=np.column_stack((df["bin_idx"].to_numpy(dtype=np.int32), df["barcode"].astype(str).to_numpy())),
            hovertext=_build_metric_hover(df, color_by),
            hovertemplate="%{hovertext}<extra></extra>",
            marker=marker_kwargs,
        )
    )


def _build_overlay_figure(
    *,
    trace,
    step_state,
    color_by: str,
    mode: str,
    uirevision_key: str,
) -> go.Figure:
    fig = go.Figure()
    bins_df = step_state.bin_table
    gt_outline_xy = None
    gt_boundary_bins_xy = None

    if trace.gt_cell_xy_um is not None:
        outline = build_gt_outline_segments(trace.gt_cell_xy_um)
        if outline is not None:
            gt_outline_xy = outline
        boundary_bins = build_gt_boundary_bin_centers(trace.gt_cell_xy_um)
        if boundary_bins is not None:
            gt_boundary_bins_xy = boundary_bins

    unassigned_non_frontier = bins_df.loc[~bins_df["is_assigned"] & ~bins_df["is_frontier"]].copy()
    frontier_df = bins_df.loc[bins_df["is_frontier"]].copy()
    assigned_df = bins_df.loc[bins_df["is_assigned"]].copy()

    _add_bin_trace(
        fig,
        df=unassigned_non_frontier,
        name="candidate bins",
        color_by=color_by,
        marker_kwargs={
            "size": 5,
            "color": "rgba(140, 140, 140, 0.15)",
            "line": {"width": 0},
        },
    )

    if not frontier_df.empty:
        if mode == "metric":
            frontier_values = frontier_df[color_by].to_numpy(dtype=np.float64)
            frontier_marker = {
                "size": 8,
                "line": {"width": 0.6, "color": "rgba(40, 40, 40, 0.15)"},
                "color": frontier_values,
                "colorbar": {
                    "title": color_by,
                    "len": 0.72,
                    "thickness": 14,
                    "x": 1.04,
                    "xanchor": "left",
                    "y": 0.5,
                    "yanchor": "middle",
                },
            }
            frontier_marker.update(_series_color_kwargs(color_by, frontier_values))
        else:
            frontier_marker = {
                "size": 8,
                "color": "#4C78A8",
                "line": {"width": 0.5, "color": "rgba(30, 30, 30, 0.20)"},
            }
        _add_bin_trace(
            fig,
            df=frontier_df,
            name="frontier bins",
            color_by=color_by,
            marker_kwargs=frontier_marker,
        )

    _add_bin_trace(
        fig,
        df=assigned_df,
        name="assigned bins",
        color_by=color_by,
        marker_kwargs={
            "size": 8,
            "color": "#E63946",
            "line": {"width": 0},
        },
    )

    if step_state.chosen_action > 0:
        chosen_df = bins_df.loc[bins_df["bin_idx"] == (step_state.chosen_action - 1)]
        _add_bin_trace(
            fig,
            df=chosen_df,
            name="greedy choice",
            color_by=color_by,
            marker_kwargs={
                "size": 14,
                "symbol": "star",
                "color": "#F4A261",
                "line": {"width": 1.2, "color": "#7F5539"},
            },
        )

    if gt_outline_xy is not None:
        x_coords, y_coords = gt_outline_xy
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line={"color": "#000000", "width": 4.8},
                hoverinfo="skip",
                name="GT cell outline",
            )
        )

    if gt_boundary_bins_xy is not None:
        fig.add_trace(
            go.Scattergl(
                x=gt_boundary_bins_xy[:, 0],
                y=gt_boundary_bins_xy[:, 1],
                mode="markers",
                name="GT boundary bins",
                marker={
                    "size": 12,
                    "symbol": "square-open",
                    "color": "#000000",
                    "line": {"width": 2.2, "color": "#000000"},
                },
                hoverinfo="skip",
            )
        )

    center = np.asarray(trace.nucleus_center_xy_um, dtype=np.float32)
    if center.shape == (2,):
        fig.add_trace(
            go.Scatter(
                x=[float(center[0])],
                y=[float(center[1])],
                mode="markers",
                name="nucleus center",
                marker={
                    "size": 16,
                    "symbol": "x",
                    "color": "#1D3557",
                    "line": {"width": 2.0, "color": "#1D3557"},
                },
                hoverinfo="skip",
            )
        )

    metrics = trace.episode_metrics
    title_lines = [
        f"cell={trace.cell_id}  step={step_state.step_index}/{max(len(trace.step_states) - 1, 0)}",
        (
            f"eval reward={float(metrics.get('total_reward', np.nan)):.2f}  "
            f"eval assigned={int(metrics.get('n_assigned_bins', 0))}/{int(metrics.get('n_candidate_bins', 0))}"
        ),
    ]
    match_method = str(metrics.get("match_method", "unmatched"))
    iou = metrics.get("pred_iou", np.nan)
    dice = metrics.get("pred_dice", np.nan)
    if pd.notna(iou) and pd.notna(dice):
        title_lines.append(f"GT match={match_method}  IoU={float(iou):.3f}  Dice={float(dice):.3f}")
    else:
        title_lines.append(f"GT match={match_method}")

    fig.update_layout(
        title="<br>".join(title_lines),
        xaxis_title="x (um)",
        yaxis_title="y (um)",
        legend_title_text="overlay",
        uirevision=uirevision_key,
        height=920,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "left",
            "x": 0.0,
        },
        dragmode="pan",
        hovermode="closest",
        template="simple_white",
        margin={"l": 20, "r": 90 if mode == "metric" else 20, "t": 90, "b": 125},
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1.0, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def _make_feature_df(values: np.ndarray, labels: tuple[str, ...]) -> pd.DataFrame:
    arr = np.asarray(values, dtype=np.float32)
    return pd.DataFrame({"feature": list(labels), "value": arr.astype(float)})


def _render_info_header(title: str, body: str) -> None:
    title_col, info_col = st.columns((24, 1), gap="small")
    title_col.markdown(f"**{title}**")
    with info_col:
        with st.popover("ⓘ"):
            st.write(body)


def _render_metric_block(title: str, metrics: dict[str, float]) -> None:
    _render_info_header(title, SECTION_HELP.get(title, title))
    help_map = (
        STATE_METRIC_HELP
        if title == "State Summary"
        else REWARD_DECOMP_HELP
        if title == "Reward Decomposition"
        else POLICY_SHAPE_HELP
        if title == "Policy / Shape Diagnostics"
        else STOP_METRIC_HELP
        if title == "STOP Panel"
        else {}
    )
    cols = st.columns(2)
    items = list(metrics.items())
    for idx, (key, value) in enumerate(items):
        cols[idx % 2].metric(key, f"{float(value):.4f}", help=help_map.get(key))


def _install_step_hotkeys() -> None:
    """Install parent-document keyboard shortcuts for step navigation."""
    components.html(
        """
        <script>
        (function() {
          const parentDoc = window.parent && window.parent.document;
          if (!parentDoc) {
            return;
          }

          function isEditableTarget(target) {
            if (!target) return false;
            const tag = (target.tagName || "").toUpperCase();
            return (
              tag === "INPUT" ||
              tag === "TEXTAREA" ||
              tag === "SELECT" ||
              target.isContentEditable
            );
          }

          function clickButtonByLabel(label) {
            const buttons = Array.from(parentDoc.querySelectorAll("button"));
            const hit = buttons.find((btn) => (btn.innerText || "").trim() === label);
            if (hit) {
              hit.click();
              return true;
            }
            return false;
          }

          if (!window.parent.__ppoDebugStepHotkeysInstalled) {
            parentDoc.addEventListener("keydown", function(event) {
              if (isEditableTarget(event.target)) {
                return;
              }
              const key = event.key;
              let handled = false;
              if (key === "ArrowLeft" || key === "a" || key === "A") {
                handled = clickButtonByLabel("Prev Step");
              } else if (key === "ArrowRight" || key === "d" || key === "D") {
                handled = clickButtonByLabel("Next Step");
              } else if (key === "Home") {
                handled = clickButtonByLabel("First Step");
              } else if (key === "End") {
                handled = clickButtonByLabel("Last Step");
              }
              if (handled) {
                event.preventDefault();
                event.stopPropagation();
              }
            }, true);
            window.parent.__ppoDebugStepHotkeysInstalled = true;
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="PPO Debug App", layout="wide")
    _inject_app_styles()

    st.title("PPO Step Debugger")
    st.caption("Per-bin reward and policy diagnostics with exact saved-trace replay when available, otherwise greedy fallback.")

    with st.sidebar:
        st.header("Inputs")
        default_run_dir = args.ppo_eval_run_dir or "runs/human_colorectal_ppo_eval_20260421T204714Z"
        ppo_eval_run_dir = st.text_input("PPO eval run dir", value=default_run_dir)
        checkpoint_override = st.text_input("Checkpoint override", value=args.checkpoint or "")
        device = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda"],
            index=["auto", "cpu", "cuda"].index(args.device),
        )
        color_by = st.selectbox("Color frontier by", options=list(COLOR_OPTIONS), index=0)

    if not ppo_eval_run_dir.strip():
        st.info("Provide a PPO eval run directory to start.")
        st.stop()

    try:
        session = _load_session(
            ppo_eval_run_dir=str(ppo_eval_run_dir).strip(),
            checkpoint=checkpoint_override.strip() or None,
            device=str(device),
            session_schema_version=DEBUG_SESSION_SCHEMA_VERSION,
        )
        if not hasattr(session, "get_bin_expression_table"):
            _load_session.clear()
            session = _load_session(
                ppo_eval_run_dir=str(ppo_eval_run_dir).strip(),
                checkpoint=checkpoint_override.strip() or None,
                device=str(device),
                session_schema_version=DEBUG_SESSION_SCHEMA_VERSION + "::reloaded",
            )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    eval_df = session.per_episode_df.copy()
    eval_df["cell_label"] = eval_df.apply(
        lambda row: (
            f"{row['cell_id']} | reward={float(row.get('total_reward', np.nan)):.2f}"
            f" | IoU={float(row.get('pred_iou', np.nan)):.3f}" if pd.notna(row.get("pred_iou", np.nan))
            else f"{row['cell_id']} | reward={float(row.get('total_reward', np.nan)):.2f}"
        ),
        axis=1,
    )
    cell_options = eval_df["cell_id"].astype(str).tolist()
    default_cell = args.cell_id if args.cell_id in cell_options else cell_options[0]
    selected_cell = st.selectbox(
        "Cell",
        options=cell_options,
        index=cell_options.index(default_cell),
        format_func=lambda cell_id: eval_df.loc[eval_df["cell_id"] == cell_id, "cell_label"].iloc[0],
    )

    try:
        trace = session.get_trace(selected_cell)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not trace.step_states:
        st.error(f"Cell {selected_cell} produced no decision states.")
        st.stop()

    cell_context = (str(session.debug_summary.eval_run_dir), str(selected_cell))
    step_state_key = f"ppo_debug_step_index::{selected_cell}"
    pending_step_key = f"ppo_debug_pending_step_index::{selected_cell}"
    autoplay_key = f"ppo_debug_autoplay_active::{selected_cell}"
    autoplay_delay_key = "ppo_debug_autoplay_delay_seconds"
    autoplay_delay_widget_key = "ppo_debug_autoplay_delay_choice"
    autoplay_delay_labels = ("0.15s", "0.25s", "0.35s", "0.50s", "0.75s", "1.00s")
    autoplay_delay_map = {
        "0.15s": 0.15,
        "0.25s": 0.25,
        "0.35s": 0.35,
        "0.50s": 0.50,
        "0.75s": 0.75,
        "1.00s": 1.00,
    }
    delay_to_label = {v: k for k, v in autoplay_delay_map.items()}
    if st.session_state.get("ppo_debug_active_cell_context") != cell_context:
        st.session_state["ppo_debug_active_cell_context"] = cell_context
        st.session_state[step_state_key] = 0
        st.session_state[pending_step_key] = None
        st.session_state[autoplay_key] = False

    max_step_index = len(trace.step_states) - 1
    if step_state_key not in st.session_state:
        st.session_state[step_state_key] = 0
    if pending_step_key not in st.session_state:
        st.session_state[pending_step_key] = None
    if autoplay_key not in st.session_state:
        st.session_state[autoplay_key] = False
    if autoplay_delay_key not in st.session_state:
        st.session_state[autoplay_delay_key] = 0.35
    if float(st.session_state[autoplay_delay_key]) not in delay_to_label:
        st.session_state[autoplay_delay_key] = 0.35
    if autoplay_delay_widget_key not in st.session_state:
        st.session_state[autoplay_delay_widget_key] = delay_to_label[float(st.session_state[autoplay_delay_key])]

    pending_step_index = st.session_state.get(pending_step_key)
    if pending_step_index is not None:
        st.session_state[step_state_key] = max(0, min(int(pending_step_index), max_step_index))
        st.session_state[pending_step_key] = None
    st.session_state[step_state_key] = max(0, min(int(st.session_state[step_state_key]), max_step_index))

    nav_cols = st.columns((1, 1, 5, 1, 1))
    with nav_cols[0]:
        if st.button("First Step", key=f"first_step::{selected_cell}", disabled=st.session_state[step_state_key] <= 0):
            st.session_state[pending_step_key] = 0
            st.rerun()
    with nav_cols[1]:
        if st.button("Prev Step", key=f"prev_step::{selected_cell}", disabled=st.session_state[step_state_key] <= 0):
            st.session_state[pending_step_key] = max(0, int(st.session_state[step_state_key]) - 1)
            st.rerun()
    with nav_cols[2]:
        step_index = st.slider(
            "Decision step",
            min_value=0,
            max_value=max_step_index,
            value=int(st.session_state[step_state_key]),
            step=1,
            key=step_state_key,
        )
    with nav_cols[3]:
        if st.button(
            "Next Step",
            key=f"next_step::{selected_cell}",
            disabled=st.session_state[step_state_key] >= max_step_index,
        ):
            st.session_state[pending_step_key] = min(max_step_index, int(st.session_state[step_state_key]) + 1)
            st.rerun()
    with nav_cols[4]:
        if st.button(
            "Last Step",
            key=f"last_step::{selected_cell}",
            disabled=st.session_state[step_state_key] >= max_step_index,
        ):
            st.session_state[pending_step_key] = max_step_index
            st.rerun()

    step_index = int(st.session_state[step_state_key])
    if st.session_state[autoplay_key] and step_index >= max_step_index:
        st.session_state[autoplay_key] = False
    _install_step_hotkeys()
    st.caption("Hotkeys: Left/Right or A/D for prev/next step, Home/End for first/last.")

    play_cols = st.columns((1, 1, 2, 3))
    with play_cols[0]:
        if st.button(
            "Pause" if st.session_state[autoplay_key] else "Play",
            key=f"play_pause::{selected_cell}",
        ):
            if st.session_state[autoplay_key]:
                st.session_state[autoplay_key] = False
            else:
                if step_index >= max_step_index:
                    st.session_state[pending_step_key] = 0
                st.session_state[autoplay_key] = True
            st.rerun()
    with play_cols[1]:
        if st.button("Stop", key=f"stop_play::{selected_cell}"):
            st.session_state[autoplay_key] = False
            st.session_state[pending_step_key] = 0
            st.rerun()
    with play_cols[2]:
        st.select_slider(
            "Play speed",
            options=list(autoplay_delay_labels),
            key=autoplay_delay_widget_key,
        )
        st.session_state[autoplay_delay_key] = autoplay_delay_map[str(st.session_state[autoplay_delay_widget_key])]
    with play_cols[3]:
        st.caption("Play animates the add process from the current step to the last step.")

    step_state = trace.step_states[step_index]

    _render_info_header(
        "Replay Summary",
        "\n\n".join(f"{k}: {v}" for k, v in REPLAY_METRIC_HELP.items()),
    )
    header_cols = st.columns(5)
    header_cols[0].metric("Replay Steps", f"{trace.n_steps_replayed}", help=REPLAY_METRIC_HELP["Replay Steps"])
    header_cols[1].metric("Replay Assigned", f"{trace.n_assigned_bins_replayed}", help=REPLAY_METRIC_HELP["Replay Assigned"])
    header_cols[2].metric("Replay Reward", f"{trace.total_reward_replayed:.3f}", help=REPLAY_METRIC_HELP["Replay Reward"])
    header_cols[3].metric("Replay Source", str(trace.replay_source), help=REPLAY_METRIC_HELP["Replay Source"])
    header_cols[4].metric("Replay Match", "yes" if trace.replay_matches_eval else "no", help=REPLAY_METRIC_HELP["Replay Match"])
    if not trace.replay_matches_eval:
        st.warning(
            "Replay does not exactly match eval row metrics. Re-run evaluate_ppo_checkpoint with saved step traces if you need exact one-to-one path replay."
        )

    left_col, right_col = st.columns((3.6, 1.0), gap="medium")

    inspect_bin_key = f"ppo_debug_inspect_bin::{selected_cell}"
    inspect_options = step_state.bin_table["bin_idx"].astype(int).tolist()
    default_inspect_bin = int(step_state.chosen_action - 1) if step_state.chosen_action > 0 else int(inspect_options[0])
    if inspect_bin_key not in st.session_state or int(st.session_state[inspect_bin_key]) not in inspect_options:
        st.session_state[inspect_bin_key] = default_inspect_bin

    frontier_table = (
        step_state.bin_table.loc[step_state.bin_table["is_frontier"]]
        .sort_values(["policy_probability", "reward_total"], ascending=[False, False])
        .head(20)
        .loc[
            :,
            [
                "bin_idx",
                "barcode",
                "policy_probability",
                "policy_logit",
                "reward_total",
                "reward_rank",
                "probability_rank",
                "expr_weighted",
                "expr_old_weighted",
                "distance_penalty",
                "overlap_penalty",
                "neighbor_bonus",
            ],
        ]
    )

    inspect_bin_idx = int(
        st.session_state[inspect_bin_key]
    )
    expr_table, expr_summary = session.get_bin_expression_table(
        cell_id=str(selected_cell),
        bin_idx=inspect_bin_idx,
        top_k=25,
    )

    with left_col:
        st.caption("Hover any bin to see barcode, frontier/assigned status, reward, and policy probability.")
        structure_tab, metric_tab = st.tabs(["Structure Overlay", "Metric Overlay"])
        with structure_tab:
            structure_fig = _build_overlay_figure(
                trace=trace,
                step_state=step_state,
                color_by=color_by,
                mode="structure",
                uirevision_key=f"{selected_cell}::structure",
            )
            st.plotly_chart(
                structure_fig,
                use_container_width=True,
                key=f"ppo_debug_structure_plot::{selected_cell}",
                on_select="ignore",
            )

        with metric_tab:
            metric_fig = _build_overlay_figure(
                trace=trace,
                step_state=step_state,
                color_by=color_by,
                mode="metric",
                uirevision_key=f"{selected_cell}::metric::{color_by}",
            )
            st.plotly_chart(
                metric_fig,
                use_container_width=True,
                key=f"ppo_debug_metric_plot::{selected_cell}",
                on_select="ignore",
            )

    with right_col:
        _render_info_header("Inspect Bin Expression", BIN_EXPRESSION_HELP)
        inspect_bin_idx = int(
            st.number_input(
                "Bin",
                min_value=int(min(inspect_options)),
                max_value=int(max(inspect_options)),
                step=1,
                key=inspect_bin_key,
            )
        )
        inspect_row = step_state.bin_table.loc[step_state.bin_table["bin_idx"] == inspect_bin_idx].iloc[0]
        st.caption(
            f"barcode={inspect_row['barcode']} | frontier={bool(inspect_row['is_frontier'])} | assigned={bool(inspect_row['is_assigned'])}"
        )
        expr_table, expr_summary = session.get_bin_expression_table(
            cell_id=str(selected_cell),
            bin_idx=inspect_bin_idx,
            top_k=25,
        )
        expr_sum_cols = st.columns(2)
        expr_sum_cols[0].metric("expr_sum", f"{float(expr_summary['sum']):.3f}")
        expr_sum_cols[1].metric("expr_max", f"{float(expr_summary['max']):.3f}")
        expr_sum_cols[0].metric("nonzero_genes", f"{int(expr_summary['nonzero_genes'])}")
        expr_sum_cols[1].metric("n_genes", f"{int(expr_summary['n_genes'])}")
        st.markdown("**Top Genes**")
        st.dataframe(expr_table, use_container_width=True, hide_index=True)
        with st.expander("Full Expression Vector", expanded=False):
            full_expr_table, _ = session.get_bin_expression_table(
                cell_id=str(selected_cell),
                bin_idx=inspect_bin_idx,
                top_k=-1,
            )
            st.dataframe(full_expr_table, use_container_width=True, hide_index=True)

    detail_left, detail_right = st.columns((1.35, 1.0), gap="medium")

    with detail_left:
        state_summary = dict(step_state.state_summary)
        state_summary["value_estimate"] = float(step_state.value_estimate)
        _render_metric_block(
            "State Summary",
            {
                "assigned_frac": state_summary["assigned_frac"],
                "step_frac": state_summary["step_frac"],
                "remaining_frac": state_summary["remaining_frac"],
                "grow_ratio_scaled": state_summary["grow_ratio_scaled"],
                "positive_frontier_fraction": state_summary["positive_frontier_fraction"],
                "centroid_drift_scaled": state_summary["centroid_drift_scaled"],
                "compactness_proxy": state_summary["compactness_proxy"],
                "frontier_add_reward_topk_mean": state_summary["frontier_add_reward_topk_mean"],
                "frontier_add_reward_mean": state_summary["frontier_add_reward_mean"],
                "frontier_add_reward_std": state_summary["frontier_add_reward_std"],
                "frontier_add_reward_max": state_summary["frontier_add_reward_max"],
                "seed_compactness": state_summary["seed_compactness"],
                "seed_radius_p90_scaled": state_summary["seed_radius_p90_scaled"],
                "seed_aspect_ratio_scaled": state_summary["seed_aspect_ratio_scaled"],
                "value_estimate": state_summary["value_estimate"],
            },
        )

        _render_info_header("STOP Panel", SECTION_HELP["STOP Panel"])
        stop_cols = st.columns(2)
        stop_cols[0].metric("stop_prob", f"{step_state.stop_probability:.4f}", help=STOP_METRIC_HELP["stop_prob"])
        stop_cols[1].metric("stop_reward", f"{step_state.stop_reward:.4f}", help=STOP_METRIC_HELP["stop_reward"])
        stop_cols[0].metric("stop_logit", f"{step_state.stop_logit:.4f}", help=STOP_METRIC_HELP["stop_logit"])
        stop_cols[1].metric("stop_delta", f"{step_state.stop_delta:.4f}", help=STOP_METRIC_HELP["stop_delta"])
        st.caption(
            f"stop_stat={session.config.stop_stat} | stop_top_k={int(session.config.stop_top_k)} | "
            f"stop_lambda={float(session.config.stop_lambda):.4f}"
        )
        if trace.step_trace_path is not None:
            st.caption(f"saved_step_trace={trace.step_trace_path}")

        if step_state.chosen_action == 0:
            st.markdown("**Greedy Action**")
            st.info("Greedy policy chooses STOP at this step.")
        else:
            chosen_row = step_state.bin_table.loc[step_state.bin_table["bin_idx"] == (step_state.chosen_action - 1)].iloc[0]
            st.markdown("**Greedy Action**")
            chosen_cols = st.columns(2)
            chosen_cols[0].metric("chosen_bin_idx", f"{int(chosen_row['bin_idx'])}")
            chosen_cols[1].metric("chosen_prob", f"{step_state.chosen_action_probability:.4f}")
            chosen_cols[0].metric("chosen_reward", f"{step_state.chosen_reward:.4f}")
            chosen_cols[1].metric("chosen_logit", f"{step_state.chosen_action_logit:.4f}")
            st.caption(f"barcode={chosen_row['barcode']}")
            st.write(
                {
                    "bin_idx": int(chosen_row["bin_idx"]),
                    "barcode": str(chosen_row["barcode"]),
                    "is_assigned": bool(chosen_row["is_assigned"]),
                    "is_frontier": bool(chosen_row["is_frontier"]),
                    "policy_rank": None
                    if pd.isna(chosen_row["probability_rank"])
                    else int(chosen_row["probability_rank"]),
                    "reward_rank": None if pd.isna(chosen_row["reward_rank"]) else int(chosen_row["reward_rank"]),
                }
            )

            _render_metric_block(
                "Reward Decomposition",
                {
                    "reward_total": float(chosen_row["reward_total"]),
                    "expr_raw": float(chosen_row["expr_raw"]),
                    "expr_conf": float(chosen_row["expr_confidence"]),
                    "expr_term": float(chosen_row["expr_term"]),
                    "w1_expr": float(chosen_row["expr_weighted"]),
                    "expr_old_raw": float(chosen_row["expr_old_raw"]),
                    "expr_old_term": float(chosen_row["expr_old_term"]),
                    "w5_expr_old": float(chosen_row["expr_old_weighted"]),
                    "w2_p_dis": float(chosen_row["distance_penalty"]),
                    "w3_p_overlap": float(chosen_row["overlap_penalty"]),
                    "w4_neighbor": float(chosen_row["neighbor_bonus"]),
                    "neighbor_support": float(chosen_row["neighbor_support"]),
                },
            )
            _render_metric_block(
                "Policy / Shape Diagnostics",
                {
                    "policy_prob": float(chosen_row["policy_probability"]),
                    "policy_logit": float(chosen_row["policy_logit"]),
                    "centroid_distance": float(chosen_row["candidate_to_current_centroid_distance"]),
                    "compactness_gain": float(chosen_row["candidate_compactness_gain"]),
                    "expr_frontier_mean": float(step_state.expr_frontier_mean),
                    "expr_frontier_std": float(step_state.expr_frontier_std),
                },
            )

            with st.expander("Feature Vectors", expanded=False):
                st.markdown("**Global Features**")
                st.dataframe(
                    _make_feature_df(step_state.global_features, GLOBAL_FEATURE_NAMES),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**Chosen ADD Action Features**")
                st.dataframe(
                    _make_feature_df(
                        step_state.action_features[int(chosen_row["bin_idx"]) + 1],
                        ADD_ACTION_FEATURE_LABELS,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**STOP Action Features**")
                st.dataframe(
                    _make_feature_df(step_state.action_features[0], STOP_ACTION_FEATURE_LABELS),
                    use_container_width=True,
                    hide_index=True,
                )

        if step_state.chosen_action == 0:
            with st.expander("Feature Vectors", expanded=False):
                st.markdown("**Global Features**")
                st.dataframe(
                    _make_feature_df(step_state.global_features, GLOBAL_FEATURE_NAMES),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**STOP Action Features**")
                st.dataframe(
                    _make_feature_df(step_state.action_features[0], STOP_ACTION_FEATURE_LABELS),
                    use_container_width=True,
                    hide_index=True,
                )

    with detail_right:
        st.markdown("**Top Frontier Bins**")
        st.dataframe(frontier_table, use_container_width=True, hide_index=True)

        with st.expander("Trajectory Table", expanded=False):
            traj_rows: list[dict[str, Any]] = []
            for s in trace.step_states:
                traj_rows.append(
                    {
                        "step_index": int(s.step_index),
                        "chosen_action": int(s.chosen_action),
                        "chosen_barcode": s.chosen_barcode,
                        "chosen_reward": float(s.chosen_reward),
                        "chosen_prob": float(s.chosen_action_probability),
                        "stop_prob": float(s.stop_probability),
                        "n_assigned_bins_after": int(s.n_assigned_bins_after),
                        "terminated_after_action": bool(s.terminated_after_action),
                        "truncated_after_action": bool(s.truncated_after_action),
                    }
                )
            st.dataframe(pd.DataFrame(traj_rows), use_container_width=True, hide_index=True)

    if st.session_state[autoplay_key]:
        if step_index >= max_step_index:
            st.session_state[autoplay_key] = False
        else:
            time.sleep(float(st.session_state[autoplay_delay_key]))
            st.session_state[pending_step_key] = int(step_index) + 1
            st.rerun()


if __name__ == "__main__":
    main()
