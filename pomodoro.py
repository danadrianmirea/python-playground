#!/usr/bin/env python3
"""
pomodoro.py — Terminal pomodoro timer with session tracking and stats.

Usage:
    pomodoro start [--task T] [--pomodoros N]     Start a pomodoro session
    pomodoro stats                                  Show focus statistics
    pomodoro log [--days N]                         Show recent session log
    pomodoro config [--work M] [--break M] [...]    View or change defaults

Examples:
    python pomodoro.py start
    python pomodoro.py start --task "Review PR" --pomodoros 6
    python pomodoro.py stats
    python pomodoro.py config --work 30 --break 10

Requires:  rich   (pip install rich)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Fix Unicode rendering on Windows terminals.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".pomodoro_config.json"
SESSIONS_PATH = Path.home() / ".pomodoro_sessions.json"

DEFAULT_WORK = 25
DEFAULT_BREAK = 5
DEFAULT_LONG_BREAK = 15
DEFAULT_INTERVALS = 4

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    work_duration: int = DEFAULT_WORK
    break_duration: int = DEFAULT_BREAK
    long_break_duration: int = DEFAULT_LONG_BREAK
    intervals_before_long_break: int = DEFAULT_INTERVALS

    @classmethod
    def load(cls) -> Config:
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                valid_keys = cls.__dataclass_fields__.keys()
                kwargs = {k: int(data[k]) for k in valid_keys if k in data}
                return cls(**kwargs)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Session log
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    date: str           # YYYY-MM-DD
    start_time: str     # HH:MM
    task: str
    pomodoros: int      # completed work intervals
    total_work_minutes: int


@dataclass
class SessionsLog:
    sessions: list[SessionRecord] = field(default_factory=list)

    @classmethod
    def load(cls) -> SessionsLog:
        if SESSIONS_PATH.exists():
            try:
                data = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
                return cls(
                    sessions=[SessionRecord(**s) for s in data.get("sessions", [])]
                )
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        return cls()

    def save(self) -> None:
        SESSIONS_PATH.write_text(
            json.dumps({"sessions": [asdict(s) for s in self.sessions]}, indent=2),
            encoding="utf-8",
        )

    def add(self, record: SessionRecord) -> None:
        self.sessions.append(record)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

console = Console(force_terminal=True)


def _fmt_minutes(m: int) -> str:
    h, r = divmod(m, 60)
    return f"{h}h {r:02d}m" if h else f"{r}m"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _now_str() -> str:
    return datetime.now().strftime("%H:%M")


# ---------------------------------------------------------------------------
# Timer display
# ---------------------------------------------------------------------------

def _build_phase_display(
    phase_name: str,
    remaining_secs: int,
    total_secs: int,
    pomodoro_num: int,
    target_pomodoros: int,
    task: str,
) -> Panel:
    """Rich Panel showing the live countdown with progress bar."""
    minutes, seconds = divmod(remaining_secs, 60)
    total_minutes = total_secs // 60
    progress = 1 - (remaining_secs / total_secs) if total_secs > 0 else 0

    is_work = phase_name == "WORK"
    icon = "🍅" if is_work else "☕"
    border_style = "red" if is_work else ("blue" if "LONG" in phase_name else "green")

    # Text-based progress bar
    bar_width = 34
    filled = int(progress * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    time_str = f"{minutes:02d}:{seconds:02d} / {total_minutes:02d}:00"

    lines: list[str] = []
    lines.append("")
    lines.append(f"  {bar}  {time_str}  ")
    lines.append("")
    if task:
        lines.append(f"  Task: {task}  ")
    lines.append("")
    lines.append("  Ctrl+C → Pause / Skip / Quit  ")

    content = Text("\n".join(lines))
    title = f" {icon}  {phase_name}  {pomodoro_num}/{target_pomodoros} "

    return Panel(content, title=title, border_style=border_style)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def notify(phase_name: str, duration_name: str) -> None:
    """Flash the terminal and beep when a phase completes."""
    is_work = phase_name == "WORK"
    icon = "🍅" if is_work else "☕"

    console.print()
    console.print(Panel(
        f"[bold]{icon}  {phase_name} complete!  Time for {duration_name}.[/]",
        border_style="yellow",
    ))

    # System beep
    print("\a", end="", flush=True)

    # Windows-specific: beep + flash taskbar
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.Beep(880, 150)
            ctypes.windll.user32.FlashWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), True
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

async def _run_phase(
    phase_name: str,
    duration_minutes: int,
    pomodoro_num: int,
    target_pomodoros: int,
    task: str,
) -> bool:
    """
    Run a single countdown phase.

    Returns True if the phase completed naturally, False if skipped.
    Raises KeyboardInterrupt if the user quits.
    """
    duration_secs = duration_minutes * 60
    elapsed = 0

    while elapsed < duration_secs:
        remaining = duration_secs - elapsed
        renderable = _build_phase_display(
            phase_name, remaining, duration_secs,
            pomodoro_num, target_pomodoros, task,
        )
        try:
            with Live(renderable, refresh_per_second=2, console=console) as live:
                while elapsed < duration_secs:
                    await asyncio.sleep(1)
                    elapsed += 1
                    live.update(_build_phase_display(
                        phase_name,
                        duration_secs - elapsed,
                        duration_secs,
                        pomodoro_num,
                        target_pomodoros,
                        task,
                    ))
        except KeyboardInterrupt:
            action = console.input(
                "\n[bright_black](P)ause  (S)kip  (Q)uit: [/]"
            ).strip().lower()
            if action == "q":
                raise  # abort entire session
            if action == "s":
                console.print("[yellow]Skipped.[/]")
                return False
            # Pause: wait for user to resume
            console.print("[bright_black]Paused. Press Enter to resume...[/]")
            input()
            # Loop continues with elapsed unchanged

    return True


# ---------------------------------------------------------------------------
# Full session runner
# ---------------------------------------------------------------------------

async def _run_full_session(config: Config, task: str, target_pomodoros: int) -> None:
    """Run an entire pomodoro session (work + breaks)."""
    log = SessionsLog.load()

    for pomodoro_num in range(1, target_pomodoros + 1):
        # --- Work ---
        completed = await _run_phase(
            "WORK", config.work_duration,
            pomodoro_num, target_pomodoros, task,
        )
        if not completed:
            console.print("[yellow]Session ended early (phase skipped).[/]")
            break

        notify("WORK", f"a {'long ' if pomodoro_num % config.intervals_before_long_break == 0 else ''}break")

        if pomodoro_num == target_pomodoros:
            # All done — no break needed after the last pomodoro
            break

        # --- Break ---
        is_long = (pomodoro_num % config.intervals_before_long_break == 0)
        break_name = "LONG BREAK" if is_long else "BREAK"
        break_minutes = config.long_break_duration if is_long else config.break_duration

        completed = await _run_phase(
            break_name, break_minutes,
            pomodoro_num, target_pomodoros, task,
        )
        if not completed:
            console.print("[yellow]Break skipped, starting next pomodoro.[/]")

    else:
        # Session completed all pomodoros
        work_done = target_pomodoros * config.work_duration
        log.add(SessionRecord(
            date=_today(),
            start_time=_now_str(),
            task=task or "Untitled",
            pomodoros=target_pomodoros,
            total_work_minutes=work_done,
        ))
        log.save()

        console.print()
        console.print(Panel(
            f"[bold green]Session complete![/]\n"
            f"  Pomodoros: {target_pomodoros}\n"
            f"  Total focus: {_fmt_minutes(work_done)}",
            title="🍅  Done",
            border_style="green",
        ))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> None:
    config = Config.load()
    task = args.task or ""
    target = max(1, args.pomodoros)

    try:
        asyncio.run(_run_full_session(config, task, target))
    except KeyboardInterrupt:
        console.print("\n[yellow]Session cancelled.[/]")


def cmd_stats(args: argparse.Namespace) -> None:  # noqa: ARG001
    log = SessionsLog.load()
    if not log.sessions:
        console.print("[yellow]No sessions logged yet. Start one with [bold]pomodoro start[/].[/]")
        return

    today = _today()

    # Compute stats
    today_minutes = 0
    today_pomodoros = 0
    total_minutes = 0
    total_pomodoros = 0
    daily: dict[str, int] = {}  # date -> pomodoros

    for s in log.sessions:
        total_minutes += s.total_work_minutes
        total_pomodoros += s.pomodoros
        daily[s.date] = daily.get(s.date, 0) + s.pomodoros
        if s.date == today:
            today_minutes += s.total_work_minutes
            today_pomodoros += s.pomodoros

    # Compute streak (consecutive days going backwards from today)
    streak = 0
    check = datetime.strptime(today, "%Y-%m-%d").date()
    for _ in range(365):
        key = check.isoformat()
        if key in daily:
            streak += 1
            check -= timedelta(days=1)
        else:
            break

    # 7-day bar chart
    bar_lines: list[str] = []
    for offset in range(6, -1, -1):
        day = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=offset)).date()
        label = day.strftime("%a")
        count = daily.get(day.isoformat(), 0)
        bar = "█" * min(count, 20) + "░" * max(0, min(20 - count, 20))
        bar_lines.append(f"  {label} {bar} {count}")

    chart = "\n".join(bar_lines)

    console.print()
    console.print(Panel(
        f"[bold]Today:[/]         {_fmt_minutes(today_minutes)} ({today_pomodoros} pomodoros)\n"
        f"[bold]Current streak:[/] {streak} day{'s' if streak != 1 else ''}\n"
        f"[bold]All time:[/]      {_fmt_minutes(total_minutes)} ({total_pomodoros} pomodoros)\n"
        f"\n[bold]Last 7 days:[/]\n{chart}",
        title="📊  Pomodoro Stats",
        border_style="cyan",
    ))


def cmd_log(args: argparse.Namespace) -> None:
    log = SessionsLog.load()
    if not log.sessions:
        console.print("[yellow]No sessions logged yet.[/]")
        return

    days = max(1, args.days)

    # Show most recent sessions, grouped by day
    window = datetime.now() - timedelta(days=days)
    recent = [
        s for s in reversed(log.sessions)
        if datetime.strptime(s.date, "%Y-%m-%d") >= window
    ]

    if not recent:
        console.print(f"[yellow]No sessions in the last {days} day(s).[/]")
        return

    table = Table(
        title=f"[bold]Sessions — Last {days} Day(s)[/]",
        header_style="bold cyan",
        box=None,
    )
    table.add_column("Date")
    table.add_column("Start")
    table.add_column("Task")
    table.add_column("Pomodoros", justify="right")
    table.add_column("Focus Time", justify="right")

    for s in recent:
        table.add_row(
            s.date,
            s.start_time,
            s.task if len(s.task) < 40 else s.task[:37] + "...",
            str(s.pomodoros),
            _fmt_minutes(s.total_work_minutes),
        )

    total_p = sum(s.pomodoros for s in recent)
    total_m = sum(s.total_work_minutes for s in recent)
    table.add_row("", "", "", "", "")  # spacer
    table.add_row(
        "[bold]Total[/]",
        "",
        f"{len(recent)} session{'s' if len(recent) != 1 else ''}",
        f"[bold]{total_p}[/]",
        f"[bold]{_fmt_minutes(total_m)}[/]",
    )

    console.print()
    console.print(table)


def cmd_config(args: argparse.Namespace) -> None:
    config = Config.load()
    changed = False

    if args.work is not None:
        if args.work < 1:
            console.print("[red]Work duration must be at least 1 minute.[/]")
            sys.exit(1)
        config.work_duration = args.work
        changed = True

    if args.break_dur is not None:
        if args.break_dur < 1:
            console.print("[red]Break duration must be at least 1 minute.[/]")
            sys.exit(1)
        config.break_duration = args.break_dur
        changed = True

    if args.long_break is not None:
        if args.long_break < 1:
            console.print("[red]Long break duration must be at least 1 minute.[/]")
            sys.exit(1)
        config.long_break_duration = args.long_break
        changed = True

    if args.intervals is not None:
        if args.intervals < 1:
            console.print("[red]Intervals must be at least 1.[/]")
            sys.exit(1)
        config.intervals_before_long_break = args.intervals
        changed = True

    if changed:
        config.save()
        console.print("[green]Configuration updated.[/]")

    console.print()
    console.print(Panel(
        f"[bold]Work duration:[/]               {config.work_duration} min\n"
        f"[bold]Break duration:[/]              {config.break_duration} min\n"
        f"[bold]Long break duration:[/]         {config.long_break_duration} min\n"
        f"[bold]Intervals before long break:[/] {config.intervals_before_long_break}",
        title="⚙️  Pomodoro Config",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pomodoro",
        description="Terminal pomodoro timer with session tracking.",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # ---- start ----
    p_start = sub.add_parser("start", help="Start a pomodoro session")
    p_start.add_argument("--task", "-t", default="", help="Task description")
    p_start.add_argument(
        "--pomodoros", "-p", type=int, default=4,
        help="Number of pomodoros to run (default: 4)",
    )

    # ---- stats ----
    sub.add_parser("stats", help="Show focus statistics")

    # ---- log ----
    p_log = sub.add_parser("log", help="Show recent sessions")
    p_log.add_argument(
        "--days", "-d", type=int, default=7,
        help="Show sessions from the last N days (default: 7)",
    )

    # ---- config ----
    p_cfg = sub.add_parser("config", help="View or change defaults")
    p_cfg.add_argument("--work", "-w", type=int, default=None, help="Work duration in minutes")
    p_cfg.add_argument("--break-dur", "-b", type=int, default=None, help="Break duration in minutes")
    p_cfg.add_argument("--long-break", "-l", type=int, default=None, help="Long break duration in minutes")
    p_cfg.add_argument("--intervals", "-i", type=int, default=None, help="Pomodoros before a long break")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
