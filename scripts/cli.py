#!/usr/bin/env python3
"""
Unified CLI for Poker Solver.

Provides a TUI for training, evaluation, precomputation, and run management.
"""

import signal
import sys
from pathlib import Path
from typing import Optional

import questionary
from questionary import Style

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import RankBasedBucketing
from src.abstraction.infoset import InfoSetKey
from src.game.actions import ActionType
from src.game.state import Street
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.utils.config import Config

# Custom style
custom_style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:#6C6C6C"),
        ("instruction", ""),
        ("text", ""),
    ]
)


class SolverCLI:
    """Unified CLI for poker solver operations."""

    def __init__(self):
        """Initialize CLI."""
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.checkpoint_dir = self.base_dir / "data" / "checkpoints"
        self.abstractions_dir = self.base_dir / "data" / "abstractions"
        self.current_trainer: Optional[Trainer] = None

        # Setup Ctrl+C handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n[!] Interrupt received!")

        if self.current_trainer:
            print("Saving checkpoint before exit...")
            try:
                # Get current iteration from metrics
                summary = self.current_trainer.metrics.get_summary()
                current_iter = summary.get("total_iterations", 0)

                if current_iter > 0:
                    self.current_trainer.checkpoint_manager.save(
                        self.current_trainer.solver,
                        current_iter,
                        tags=["interrupted"],
                    )
                    self.current_trainer.checkpoint_manager.update_stats(
                        total_iterations=current_iter,
                        total_runtime_seconds=self.current_trainer.metrics.get_elapsed_time(),
                        num_infosets=self.current_trainer.solver.num_infosets(),
                    )
                    print(f"[OK] Checkpoint saved at iteration {current_iter}")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint: {e}")

        print("Exiting...")
        sys.exit(0)

    def run(self):
        """Run the main TUI loop."""
        print("\n" + "=" * 60)
        print("POKER SOLVER CLI")
        print("=" * 60)

        while True:
            action = questionary.select(
                "\nWhat would you like to do?",
                choices=[
                    "Train Solver",
                    "Evaluate Solver",
                    "Precompute Equity Buckets",
                    "List Abstractions",
                    "View Past Runs",
                    "Resume Training",
                    "View Preflop Chart",
                    "Exit",
                ],
                style=custom_style,
            ).ask()

            if action is None or "Exit" in action:
                print("\nGoodbye!")
                break

            try:
                if "Train" in action and "Resume" not in action:
                    self.train_solver()
                elif "Evaluate" in action:
                    self.evaluate_solver()
                elif "Precompute" in action:
                    self.precompute_equity_buckets()
                elif "List Abstractions" in action:
                    from src.abstraction.abstraction_metadata import AbstractionManager

                    manager = AbstractionManager()
                    manager.print_summary()
                    input("\nPress Enter to continue...")
                elif "View Past" in action:
                    self.view_runs()
                elif "Resume" in action:
                    self.resume_training()
                elif "Preflop Chart" in action:
                    self.view_preflop_chart()
            except KeyboardInterrupt:
                print("\n\n[!] Operation cancelled by user")
                continue
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback

                traceback.print_exc()
                input("\nPress Enter to continue...")

    def _select_config(self) -> Optional[Config]:
        """
        Select and optionally edit a config file.

        Returns:
            Loaded Config object or None if cancelled
        """
        # List available configs
        config_files = sorted(self.config_dir.glob("*.yaml"))

        if not config_files:
            print("[ERROR] No config files found in config/")
            return None

        # Let user select
        choices = [f.stem for f in config_files] + ["Cancel"]

        selected = questionary.select(
            "Select configuration:",
            choices=choices,
            style=custom_style,
        ).ask()

        if selected == "Cancel" or selected is None:
            return None

        config_path = self.config_dir / f"{selected}.yaml"
        config = Config.from_file(config_path)

        # Ask if they want to edit
        edit = questionary.confirm(
            "Edit configuration before running?",
            default=False,
            style=custom_style,
        ).ask()

        if edit:
            config = self._edit_config(config)

        return config

    def _edit_config(self, config: Config) -> Config:
        """
        Interactive config editor.

        Args:
            config: Config to edit

        Returns:
            Modified config
        """
        print("\nEdit Configuration")
        print("-" * 40)

        # Key parameters to edit
        iterations = questionary.text(
            "Number of iterations:",
            default=str(config.get("training.num_iterations", 1000)),
            style=custom_style,
        ).ask()

        abstraction_type = questionary.select(
            "Card abstraction type:",
            choices=["rank_based", "equity_bucketing"],
            default=config.get("card_abstraction.type", "rank_based"),
            style=custom_style,
        ).ask()

        checkpoint_freq = questionary.text(
            "Checkpoint frequency:",
            default=str(config.get("training.checkpoint_frequency", 100)),
            style=custom_style,
        ).ask()

        # Update config
        config.set("training.num_iterations", int(iterations))
        config.set("card_abstraction.type", abstraction_type)
        config.set("training.checkpoint_frequency", int(checkpoint_freq))

        # If equity bucketing, check for file
        if abstraction_type == "equity_bucketing":
            default_path = "data/abstractions/equity_buckets.pkl"
            bucketing_path = questionary.text(
                "Equity bucketing file path:",
                default=config.get("card_abstraction.bucketing_path", default_path),
                style=custom_style,
            ).ask()

            config.set("card_abstraction.bucketing_path", bucketing_path)

            # Check if file exists
            if not Path(bucketing_path).exists():
                print(f"\n[!] Warning: Equity bucketing file not found: {bucketing_path}")
                precompute = questionary.confirm(
                    "Would you like to precompute equity buckets now?",
                    default=True,
                    style=custom_style,
                ).ask()

                if precompute:
                    self._run_precomputation(bucketing_path)

        return config

    def train_solver(self):
        """Train a new solver."""
        print("\nTrain Solver")
        print("=" * 60)

        config = self._select_config()
        if config is None:
            return

        # Check equity bucketing requirement
        if config.get("card_abstraction.type") == "equity_bucketing":
            bucketing_path = config.get("card_abstraction.bucketing_path")
            if bucketing_path and not Path(bucketing_path).exists():
                print(f"\n[ERROR] Equity bucketing file required but not found: {bucketing_path}")
                print("   Please precompute equity buckets first (option 3 from main menu)")
                return

        # Create trainer
        print("\nInitializing trainer...")
        trainer = Trainer(config)
        self.current_trainer = trainer

        # Start training
        print(f"\nStarting training for {config.get('training.num_iterations')} iterations...")
        print(f"Checkpoints: {trainer.checkpoint_manager.checkpoint_dir}")
        print(
            f"Checkpoint frequency: every {config.get('training.checkpoint_frequency')} iterations"
        )
        print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

        try:
            results = trainer.train()

            print("\n[OK] Training completed!")
            print(f"   Total iterations: {results['total_iterations']}")
            print(f"   Final infosets: {results['final_infosets']}")
            print(f"   Average utility: {results['avg_utility']:.4f}")
            print(f"   Elapsed time: {results['elapsed_time']:.2f}s")

        finally:
            self.current_trainer = None

        input("\nPress Enter to continue...")

    def evaluate_solver(self):
        """Evaluate a trained solver."""
        print("\nEvaluate Solver")
        print("=" * 60)

        # List available runs
        runs = CheckpointManager.list_runs(self.checkpoint_dir)

        if not runs:
            print("\n[ERROR] No trained runs found in data/checkpoints/")
            input("Press Enter to continue...")
            return

        # Select run
        selected_run = questionary.select(
            "Select run to evaluate:",
            choices=runs + ["Cancel"],
            style=custom_style,
        ).ask()

        if selected_run == "Cancel" or selected_run is None:
            return

        # TODO: Implement evaluation
        print(f"\n[!] Evaluation not yet implemented for {selected_run}")
        print("   This will compare against baseline strategies")

        input("\nPress Enter to continue...")

    def precompute_equity_buckets(self):
        """Precompute equity buckets."""
        from src.cli.precompute_handler import handle_precompute

        handle_precompute(custom_style)

    def view_runs(self):
        """View past training runs."""
        print("\nPast Training Runs")
        print("=" * 60)

        runs = CheckpointManager.list_runs(self.checkpoint_dir)

        if not runs:
            print("\n[ERROR] No training runs found")
            input("Press Enter to continue...")
            return

        # Select run to view
        selected = questionary.select(
            "Select run to view details:",
            choices=runs + ["Back"],
            style=custom_style,
        ).ask()

        if selected == "Back" or selected is None:
            return

        # Load and display run info
        manager = CheckpointManager.from_run_id(
            self.checkpoint_dir,
            selected,
        )

        if manager.run_metadata:
            meta = manager.run_metadata
            print(f"\nRun: {selected}")
            print("-" * 60)
            print(f"Status: {meta.status}")
            print(f"Started: {meta.started_at}")
            if meta.completed_at:
                print(f"Completed: {meta.completed_at}")

            if meta.statistics:
                stats = meta.statistics
                print("\nStatistics:")
                print(f"  Iterations: {stats.total_iterations}")
                print(
                    f"  Runtime: {stats.total_runtime_seconds:.2f}s ({stats.total_runtime_seconds / 60:.1f}m)"
                )
                print(f"  Speed: {stats.iterations_per_second:.2f} it/s")
                print(f"  Infosets: {stats.num_infosets:,}")

        # Show checkpoints
        if manager.manifest:
            checkpoints = manager.manifest.checkpoints
            print(f"\nCheckpoints: {len(checkpoints)}")
            for cp in checkpoints[:5]:  # Show first 5
                print(f"  {cp.iteration:6d}: {cp.num_infosets:8,} infosets {cp.tags}")
            if len(checkpoints) > 5:
                print(f"  ... and {len(checkpoints) - 5} more")

        input("\nPress Enter to continue...")

    def resume_training(self):
        """Resume training from a checkpoint."""
        print("\nResume Training")
        print("=" * 60)

        runs = CheckpointManager.list_runs(self.checkpoint_dir)

        if not runs:
            print("\n[ERROR] No training runs found to resume")
            input("Press Enter to continue...")
            return

        # Select run
        selected = questionary.select(
            "Select run to resume:",
            choices=runs + ["Cancel"],
            style=custom_style,
        ).ask()

        if selected == "Cancel" or selected is None:
            return

        # Load run config
        manager = CheckpointManager.from_run_id(self.checkpoint_dir, selected)

        if not manager.run_metadata or not manager.run_metadata.config:
            print("\n[ERROR] No config found for this run")
            input("Press Enter to continue...")
            return

        config_dict = manager.run_metadata.config
        config = Config.from_dict(config_dict)

        # Show current status
        latest = manager.get_latest_checkpoint()
        if latest:
            print(f"\nLatest checkpoint: iteration {latest['iteration']}")
            print(f"Infosets: {latest['num_infosets']:,}")

        # Ask for additional iterations
        add_iters = questionary.text(
            "Additional iterations to run:",
            default="1000",
            style=custom_style,
        ).ask()

        if add_iters is None:
            return

        total_iters = latest["iteration"] + int(add_iters)
        config.set("training.num_iterations", total_iters)

        # Create trainer with same run_id
        print("\nResuming trainer...")
        trainer = Trainer(config, run_id=selected)
        self.current_trainer = trainer

        print(f"\nResuming training from iteration {latest['iteration']}...")
        print(f"Target: {total_iters} iterations (+{add_iters})")
        print("\n[!] Press Ctrl+C to save checkpoint and exit\n")

        try:
            results = trainer.train(resume=True)

            print("\n[OK] Training completed!")
            print(f"   Total iterations: {results['total_iterations']}")
            print(f"   Final infosets: {results['final_infosets']}")

        finally:
            self.current_trainer = None

        input("\nPress Enter to continue...")

    def view_preflop_chart(self):
        """View preflop strategy chart from trained solver."""
        print("\nView Preflop Chart")
        print("=" * 60)

        # List available runs
        runs = CheckpointManager.list_runs(self.checkpoint_dir)

        if not runs:
            print("\n[ERROR] No trained runs found in data/checkpoints/")
            input("Press Enter to continue...")
            return

        # Select run
        selected_run = questionary.select(
            "Select run to visualize:",
            choices=runs + ["Cancel"],
            style=custom_style,
        ).ask()

        if selected_run == "Cancel" or selected_run is None:
            return

        # Load solver
        print(f"\nLoading solver from {selected_run}...")
        run_checkpoint_dir = self.checkpoint_dir / selected_run
        storage = DiskBackedStorage(
            checkpoint_dir=run_checkpoint_dir,
            cache_size=100000,
            flush_frequency=1000,
        )

        print(f"  Loaded {storage.num_infosets():,} infosets")

        if storage.num_infosets() == 0:
            print("\n[ERROR] No strategy data found in this run")
            input("Press Enter to continue...")
            return

        # Create solver
        action_abs = ActionAbstraction()
        card_abs = RankBasedBucketing()
        solver = MCCFRSolver(
            action_abstraction=action_abs,
            card_abstraction=card_abs,
            storage=storage,
            config={"seed": 42, "starting_stack": 200},
        )

        # Select position and situation
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

        # Generate chart
        print("\nGenerating HTML preflop chart...")
        chart_path = self._generate_html_preflop_chart(
            solver, position, betting_sequence, selected_run
        )

        print(f"\n[OK] Chart saved to: {chart_path}")
        print("     Opening in browser...")

        # Open in browser
        import webbrowser

        webbrowser.open(f"file://{chart_path.absolute()}")

        input("\nPress Enter to continue...")

    def _generate_html_preflop_chart(
        self, solver: MCCFRSolver, position: int, betting_sequence: str, run_id: str
    ) -> Path:
        """Generate HTML preflop chart with action breakdown visualization."""
        ranks = "AKQJT98765432"

        # Collect all hand strategies
        hands_data = []
        for i, r1 in enumerate(ranks):
            row = []
            for j, r2 in enumerate(ranks):
                if i == j:
                    # Pair
                    hand = f"{r1}{r2}"
                    strategy = self._get_hand_strategy(
                        solver, r1, r2, False, True, position, betting_sequence
                    )
                elif i < j:
                    # Suited (above diagonal)
                    hand = f"{r1}{r2}s"
                    strategy = self._get_hand_strategy(
                        solver, r1, r2, True, False, position, betting_sequence
                    )
                else:
                    # Offsuit (below diagonal)
                    hand = f"{r2}{r1}o"
                    strategy = self._get_hand_strategy(
                        solver, r2, r1, False, False, position, betting_sequence
                    )

                row.append({"hand": hand, "strategy": strategy})
            hands_data.append(row)

        # Generate HTML
        html = self._create_html_chart(hands_data, ranks, position, betting_sequence, run_id)

        # Save to file
        charts_dir = self.base_dir / "data" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        position_str = "btn" if position == 0 else "bb"
        situation_str = "first_to_act" if not betting_sequence else betting_sequence
        chart_path = charts_dir / f"preflop_{run_id}_{position_str}_{situation_str}.html"

        with open(chart_path, "w") as f:
            f.write(html)

        return chart_path

    def _create_html_chart(self, hands_data, ranks, position, betting_sequence, run_id) -> str:
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

        # Generate table rows
        for i, row in enumerate(hands_data):
            html += "                <tr>\n"
            for j, cell in enumerate(row):
                hand = cell["hand"]
                strategy = cell["strategy"]

                if strategy is None:
                    # No data - gray cell
                    html += f"""                    <td class="no-data" data-hand="{hand}">
                        <div class="cell-label">{hand}</div>
                    </td>
"""
                else:
                    # Has data - create gradient background
                    raise_pct = strategy["raise"] * 100
                    call_pct = strategy["call"] * 100
                    fold_pct = strategy["fold"] * 100

                    # Build CSS gradient (horizontal segments)
                    # Green for raise, blue for call, red for fold
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

        document.querySelectorAll('td').forEach(cell => {
            cell.addEventListener('mouseenter', (e) => {
                const hand = cell.dataset.hand;
                const raise = cell.dataset.raise;
                const call = cell.dataset.call;
                const fold = cell.dataset.fold;

                if (raise === undefined) {
                    tooltipHand.textContent = hand;
                    tooltipRaise.textContent = 'No data';
                    tooltipCall.textContent = 'No data';
                    tooltipFold.textContent = 'No data';
                } else {
                    tooltipHand.textContent = hand;
                    tooltipRaise.textContent = raise + '%';
                    tooltipCall.textContent = call + '%';
                    tooltipFold.textContent = fold + '%';
                }

                tooltip.classList.add('show');
            });

            cell.addEventListener('mousemove', (e) => {
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY + 15) + 'px';
            });

            cell.addEventListener('mouseleave', () => {
                tooltip.classList.remove('show');
            });
        });
    </script>
</body>
</html>
"""

        return html

    def _get_hand_strategy(
        self,
        solver: MCCFRSolver,
        rank1: str,
        rank2: str,
        suited: bool,
        is_pair: bool,
        position: int,
        betting_sequence: str,
    ) -> Optional[dict]:
        """Get strategy for a specific hand."""
        # Convert ranks to hand string (e.g., "AKs", "72o", "TT")
        hand_string = self._ranks_to_hand_string(rank1, rank2, suited, is_pair)

        # Try all SPR buckets (0=shallow, 1=medium, 2=deep)
        infoset = None
        for spr_bucket in [2, 1, 0]:  # Try deep first (200BB stacks)
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

        # Categorize actions
        result = {
            "fold": 0.0,
            "call": 0.0,
            "raise": 0.0,
        }

        for action, prob in zip(actions, strategy):
            if action.type == ActionType.FOLD:
                result["fold"] += prob
            elif action.type in [ActionType.CALL, ActionType.CHECK]:
                result["call"] += prob
            elif action.type in [ActionType.RAISE, ActionType.BET, ActionType.ALL_IN]:
                result["raise"] += prob

        return result

    def _ranks_to_hand_string(self, rank1: str, rank2: str, suited: bool, is_pair: bool) -> str:
        """Convert rank characters to hand string (e.g., AKs, 72o, TT)."""
        ranks = "AKQJT98765432"

        # For pairs
        if is_pair:
            return f"{rank1}{rank2}"

        # For non-pairs, put higher rank first
        r1_val = 14 - ranks.index(rank1)
        r2_val = 14 - ranks.index(rank2)

        if r1_val > r2_val:
            high, low = rank1, rank2
        else:
            high, low = rank2, rank1

        # Add suited/offsuit suffix
        suffix = "s" if suited else "o"

        return f"{high}{low}{suffix}"


def main():
    """Main entry point."""
    cli = SolverCLI()
    cli.run()


if __name__ == "__main__":
    main()
