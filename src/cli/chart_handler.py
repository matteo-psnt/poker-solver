"""Preflop chart generation for CLI."""

import webbrowser
from pathlib import Path
from typing import Optional

import questionary

from src.actions.betting_actions import BettingActions
from src.bucketing.base import BucketingStrategy
from src.bucketing.utils.infoset import InfoSetKey
from src.game.actions import ActionType
from src.game.state import Street
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training.run_tracker import RunTracker


def handle_view_preflop_chart(
    runs_dir: Path,
    base_dir: Path,
    custom_style,
):
    """
    Handle viewing preflop strategy chart.

    Args:
        runs_dir: Directory containing training runs
        base_dir: Base project directory
        custom_style: Questionary style
    """

    runs = RunTracker.list_runs(runs_dir)

    if not runs:
        print("\n[ERROR] No trained runs found in data/runs/")
        input("Press Enter to continue...")
        return

    selected_run = questionary.select(
        "Select run to visualize:",
        choices=runs + ["Cancel"],
        style=custom_style,
    ).ask()

    if selected_run == "Cancel" or selected_run is None:
        return

    print(f"\nLoading solver from {selected_run}...")
    run_dir = runs_dir / selected_run
    storage = InMemoryStorage(checkpoint_dir=run_dir)

    print(f"  Loaded {storage.num_infosets():,} infosets")

    if storage.num_infosets() == 0:
        print("\n[ERROR] No strategy data found in this run")
        input("Press Enter to continue...")
        return

    # For viewing charts, we don't need the full solver - just the storage
    # But MCCFRSolver requires action/card abstraction, so provide minimal instances
    action_abs = BettingActions()

    # Card abstraction not actually used for viewing stored strategies
    # Create a dummy implementation since MCCFRSolver requires it
    class DummyCardAbstraction(BucketingStrategy):
        """Placeholder abstraction for viewing charts (not used)."""

        def get_bucket(self, hole_cards, board, street):
            return 0

        def num_buckets(self, street):
            return 1

    solver = MCCFRSolver(
        action_abstraction=action_abs,
        card_abstraction=DummyCardAbstraction(),
        storage=storage,
        config={"seed": 42, "starting_stack": 200},
    )

    position_choice = questionary.select(
        "Select position:",
        choices=["Button (BTN)", "Big Blind (BB)", "Cancel"],
        style=custom_style,
    ).ask()

    if position_choice == "Cancel" or position_choice is None:
        return

    position = 0 if "Button" in position_choice else 1

    situation = questionary.select(
        "Select situation:",
        choices=["First to act", "Facing raise", "Cancel"],
        style=custom_style,
    ).ask()

    if situation == "Cancel" or situation is None:
        return

    betting_sequence = "" if "First" in situation else "r5"

    print("\nGenerating HTML preflop chart...")
    chart_path = generate_html_preflop_chart(
        solver, position, betting_sequence, selected_run, base_dir
    )

    print(f"\n[OK] Chart saved to: {chart_path}")
    print("     Opening in browser...")

    webbrowser.open(f"file://{chart_path.absolute()}")

    input("\nPress Enter to continue...")


def generate_html_preflop_chart(
    solver: MCCFRSolver,
    position: int,
    betting_sequence: str,
    run_id: str,
    base_dir: Path,
) -> Path:
    """Generate HTML preflop chart with action breakdown visualization."""
    ranks = "AKQJT98765432"

    hands_data = []
    for i, r1 in enumerate(ranks):
        row = []
        for j, r2 in enumerate(ranks):
            if i == j:
                hand = f"{r1}{r2}"
                strategy = _get_hand_strategy(
                    solver, r1, r2, False, True, position, betting_sequence
                )
            elif i < j:
                hand = f"{r1}{r2}s"
                strategy = _get_hand_strategy(
                    solver, r1, r2, True, False, position, betting_sequence
                )
            else:
                hand = f"{r2}{r1}o"
                strategy = _get_hand_strategy(
                    solver, r2, r1, False, False, position, betting_sequence
                )

            row.append({"hand": hand, "strategy": strategy})
        hands_data.append(row)

    html = _create_html_chart(hands_data, ranks, position, betting_sequence, run_id)

    charts_dir = base_dir / "data" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    position_str = "btn" if position == 0 else "bb"
    situation_str = "first_to_act" if not betting_sequence else betting_sequence
    chart_path = charts_dir / f"preflop_{run_id}_{position_str}_{situation_str}.html"

    with open(chart_path, "w") as f:
        f.write(html)

    return chart_path


def _get_hand_strategy(
    solver: MCCFRSolver,
    rank1: str,
    rank2: str,
    suited: bool,
    is_pair: bool,
    position: int,
    betting_sequence: str,
) -> Optional[dict]:
    """Get strategy for a specific hand."""
    hand_string = _ranks_to_hand_string(rank1, rank2, suited, is_pair)

    infoset = None
    for spr_bucket in [2, 1, 0]:
        key = InfoSetKey(
            player_position=position,
            street=Street.PREFLOP,
            betting_sequence=betting_sequence,
            preflop_hand=hand_string,
            postflop_bucket=None,
            spr_bucket=spr_bucket,
        )

        infoset = solver.storage.get_infoset(key)
        if infoset is not None:
            break

    if infoset is None:
        return None

    strategy = infoset.get_average_strategy()
    actions = infoset.legal_actions

    result = {"fold": 0.0, "call": 0.0, "raise": 0.0}

    for action, prob in zip(actions, strategy):
        if action.type == ActionType.FOLD:
            result["fold"] += prob
        elif action.type in [ActionType.CALL, ActionType.CHECK]:
            result["call"] += prob
        elif action.type in [ActionType.RAISE, ActionType.BET, ActionType.ALL_IN]:
            result["raise"] += prob

    return result


def _ranks_to_hand_string(rank1: str, rank2: str, suited: bool, is_pair: bool) -> str:
    """Convert rank characters to hand string."""
    ranks = "AKQJT98765432"

    if is_pair:
        return f"{rank1}{rank2}"

    r1_val = 14 - ranks.index(rank1)
    r2_val = 14 - ranks.index(rank2)

    if r1_val > r2_val:
        high, low = rank1, rank2
    else:
        high, low = rank2, rank1

    suffix = "s" if suited else "o"

    return f"{high}{low}{suffix}"


def _create_html_chart(hands_data, ranks, position, betting_sequence, run_id) -> str:
    """Create HTML with CSS for preflop chart visualization."""
    position_name = "Button (BTN)" if position == 0 else "Big Blind (BB)"
    situation = betting_sequence if betting_sequence else "first to act"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Preflop Chart - {position_name} - {situation}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            margin: 0;
            padding: 20px;
            color: #fff;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }}

        h1 {{
            color: #1e3c72;
            text-align: center;
            margin: 0 0 10px 0;
            font-size: 32px;
        }}

        .subtitle {{
            color: #666;
            text-align: center;
            margin: 0 0 30px 0;
            font-size: 16px;
        }}

        .chart-wrapper {{
            overflow-x: auto;
        }}

        table {{
            border-collapse: separate;
            border-spacing: 3px;
            margin: 0 auto;
            background: #e0e0e0;
            border-radius: 8px;
            padding: 3px;
        }}

        td {{
            width: 70px;
            height: 70px;
            position: relative;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        td:hover {{
            transform: scale(1.08);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 10;
        }}

        .cell-background {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}

        .cell-label {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 14px;
            color: #000;
            text-shadow:
                -1px -1px 0 #fff,
                1px -1px 0 #fff,
                -1px 1px 0 #fff,
                1px 1px 0 #fff,
                0 0 3px rgba(255, 255, 255, 0.8);
            pointer-events: none;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.95);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            z-index: 1000;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.2s;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }}

        .tooltip.show {{
            opacity: 1;
        }}

        .tooltip-hand {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 4px;
        }}

        .tooltip-action {{
            margin: 4px 0;
            display: flex;
            align-items: center;
        }}

        .tooltip-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
            display: inline-block;
        }}

        .legend {{
            margin: 30px auto 0;
            max-width: 600px;
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .legend h3 {{
            margin: 0 0 15px 0;
            color: #333;
            font-size: 18px;
        }}

        .legend-item {{
            display: inline-flex;
            align-items: center;
            margin: 0 20px 10px 0;
        }}

        .legend-color {{
            width: 24px;
            height: 24px;
            border-radius: 4px;
            margin-right: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}

        .legend-text {{
            color: #333;
            font-weight: 500;
        }}

        .note {{
            margin: 20px auto 0;
            max-width: 600px;
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            color: #856404;
            font-size: 13px;
            line-height: 1.6;
        }}

        .note strong {{
            display: block;
            margin-bottom: 8px;
            font-size: 14px;
        }}

        .no-data {{
            background: #9e9e9e !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Preflop Strategy Chart</h1>
        <div class="subtitle">
            {position_name} • {situation} • Run: {run_id}
        </div>

        <div class="chart-wrapper">
            <table>
"""

    for i, row in enumerate(hands_data):
        html += "                <tr>\n"
        for j, cell in enumerate(row):
            hand = cell["hand"]
            strategy = cell["strategy"]

            if strategy is None:
                html += f"""                    <td class="no-data" data-hand="{hand}">
                        <div class="cell-label">{hand}</div>
                    </td>
"""
            else:
                raise_pct = strategy["raise"] * 100
                call_pct = strategy["call"] * 100
                fold_pct = strategy["fold"] * 100

                gradient_stops = []
                current_pct = 0

                if raise_pct > 0:
                    gradient_stops.append(
                        f"#4caf50 {current_pct}%, #4caf50 {current_pct + raise_pct}%"
                    )
                    current_pct += raise_pct

                if call_pct > 0:
                    gradient_stops.append(
                        f"#2196f3 {current_pct}%, #2196f3 {current_pct + call_pct}%"
                    )
                    current_pct += call_pct

                if fold_pct > 0:
                    gradient_stops.append(
                        f"#f44336 {current_pct}%, #f44336 {current_pct + fold_pct}%"
                    )

                gradient = f"linear-gradient(to right, {', '.join(gradient_stops)})"

                html += f"""                    <td data-hand="{hand}" data-raise="{raise_pct:.1f}" data-call="{call_pct:.1f}" data-fold="{fold_pct:.1f}">
                        <div class="cell-background" style="background: {gradient};"></div>
                        <div class="cell-label">{hand}</div>
                    </td>
"""

        html += "                </tr>\n"

    html += """            </table>
        </div>

        <div class="legend">
            <h3>Action Colors</h3>
            <div class="legend-item">
                <div class="legend-color" style="background: #4caf50;"></div>
                <span class="legend-text">Raise / Bet</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2196f3;"></div>
                <span class="legend-text">Call / Check</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f44336;"></div>
                <span class="legend-text">Fold</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9e9e9e;"></div>
                <span class="legend-text">No Data</span>
            </div>
        </div>

        <div class="note">
            <strong>Chart Layout</strong>
            Pairs on diagonal (AA, KK, QQ, ...) •
            Suited hands above diagonal (AKs, AQs, ...) •
            Offsuit hands below diagonal (AKo, AQo, ...)
            <br><br>
            <strong>How to Read</strong>
            Each cell shows action percentages as color segments from left to right.
            Hover over any hand to see exact percentages.
        </div>
    </div>

    <div class="tooltip" id="tooltip">
        <div class="tooltip-hand" id="tooltip-hand"></div>
        <div class="tooltip-action">
            <span class="tooltip-color" style="background: #4caf50;"></span>
            <span>Raise: <strong id="tooltip-raise">-</strong></span>
        </div>
        <div class="tooltip-action">
            <span class="tooltip-color" style="background: #2196f3;"></span>
            <span>Call: <strong id="tooltip-call">-</strong></span>
        </div>
        <div class="tooltip-action">
            <span class="tooltip-color" style="background: #f44336;"></span>
            <span>Fold: <strong id="tooltip-fold">-</strong></span>
        </div>
    </div>

    <script>
        const tooltip = document.getElementById('tooltip');
        const tooltipHand = document.getElementById('tooltip-hand');
        const tooltipRaise = document.getElementById('tooltip-raise');
        const tooltipCall = document.getElementById('tooltip-call');
        const tooltipFold = document.getElementById('tooltip-fold');

        document.querySelectorAll('td').forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const hand = cell.dataset.hand;
                const raise = cell.dataset.raise;
                const call = cell.dataset.call;
                const fold = cell.dataset.fold;

                if (raise === undefined) {{
                    tooltipHand.textContent = hand;
                    tooltipRaise.textContent = 'No data';
                    tooltipCall.textContent = 'No data';
                    tooltipFold.textContent = 'No data';
                }} else {{
                    tooltipHand.textContent = hand;
                    tooltipRaise.textContent = raise + '%';
                    tooltipCall.textContent = call + '%';
                    tooltipFold.textContent = fold + '%';
                }}

                tooltip.classList.add('show');
            }});

            cell.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY + 15) + 'px';
            }});

            cell.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});
    </script>
</body>
</html>
"""

    return html
