#!/usr/bin/env python3
"""OpenClaw AI-analyzed floating overlay — shows insights the TUI doesn't."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import objc
import AppKit
import Foundation
import Quartz
from PyObjCTools import AppHelper

# ─── Constants ────────────────────────────────────────────────────────────────

LOG_DIR = Path("/tmp/openclaw")
SESSIONS_DIR = Path.home() / ".openclaw/agents/main/sessions"
SESSIONS_JSON = SESSIONS_DIR / "sessions.json"
GATEWAY_LOG = Path.home() / ".openclaw/logs/gateway.log"

POLL_INTERVAL = 2.0
SESSION_POLL_INTERVAL = 5.0  # faster for session JSONL
PID_POLL_INTERVAL = 20.0

WIN_W, WIN_H = 360, 260
MARGIN = 12

# Runtime log regex
RE_TOOL_START = re.compile(r"embedded run tool start: runId=(\S+) tool=(\w+)")
RE_TOOL_END = re.compile(r"embedded run tool end: runId=(\S+) tool=(\w+)")
RE_RUN_START = re.compile(r"embedded run start: runId=(\S+)")
RE_RUN_DONE = re.compile(r"embedded run done: runId=(\S+)")
RE_PROMPT_START = re.compile(r"embedded run prompt start: runId=(\S+)")
RE_PROMPT_END = re.compile(r"embedded run prompt end: runId=(\S+)")
RE_RUN_TIMEOUT = re.compile(r"embedded run timeout: runId=(\S+)")

# Colors
BG_COLOR = (0.08, 0.08, 0.10, 0.92)
COLOR_GREEN = (0.30, 0.85, 0.40, 1.0)
COLOR_YELLOW = (0.95, 0.80, 0.20, 1.0)
COLOR_GRAY = (0.50, 0.50, 0.50, 1.0)
COLOR_RED = (0.95, 0.30, 0.30, 1.0)
COLOR_DIM = (0.35, 0.35, 0.38, 1.0)
COLOR_WHITE = (0.92, 0.92, 0.94, 1.0)
COLOR_MUTED = (0.55, 0.55, 0.58, 1.0)
COLOR_SEPARATOR = (0.25, 0.25, 0.28, 1.0)
COLOR_CYAN = (0.40, 0.80, 0.95, 1.0)
COLOR_ORANGE = (0.95, 0.60, 0.20, 1.0)
COLOR_PURPLE = (0.70, 0.50, 0.90, 1.0)
COLOR_ERROR_BG = (0.22, 0.10, 0.10, 1.0)
COLOR_SECTION_BG = (0.12, 0.12, 0.14, 0.6)


# ─── AgentState ──────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    status: str = "offline"  # active, thinking, idle, error, offline
    current_tool: Optional[str] = None
    activity_start: Optional[float] = None

    # Session-derived insights (THE GOOD STUFF)
    current_task: str = ""           # what the agent is working on
    recent_tools: list = field(default_factory=list)  # last N tool calls [{name, args_summary, duration, ok}]
    total_cost: float = 0.0          # session cost in $
    cost_last_5min: float = 0.0      # burn rate
    context_tokens: int = 0
    context_max: int = 200000        # estimated
    tokens_per_min: float = 0.0      # context burn rate
    tool_counts: dict = field(default_factory=dict)  # tool -> count
    tool_errors: int = 0             # total errors this session
    tool_successes: int = 0
    retry_detected: bool = False     # same tool called 3+ times in a row
    retry_tool: str = ""
    compactions: int = 0             # how many context compactions
    model: str = ""
    gateway_pid: Optional[int] = None
    last_error: str = ""
    last_error_time: Optional[float] = None
    warnings: list = field(default_factory=list)  # AI-like warnings


# ─── SessionAnalyzer — parses the actual session JSONL for insights ──────────

class SessionAnalyzer:
    """Parses session JSONL files for deep insights the TUI doesn't show."""

    def __init__(self):
        self._session_path: Optional[Path] = None
        self._session_pos: int = 0
        self._messages: deque = deque(maxlen=500)  # rolling window
        self._last_user_msg: str = ""
        self._last_compaction_summary: str = ""
        self._tool_history: deque = deque(maxlen=50)
        self._total_cost: float = 0.0
        self._cost_timestamps: deque = deque(maxlen=100)  # (epoch, cost) for burn rate
        self._token_timestamps: deque = deque(maxlen=50)   # (epoch, tokens) for burn rate
        self._context_tokens: int = 0
        self._tool_counts: dict = {}
        self._tool_errors: int = 0
        self._tool_successes: int = 0
        self._compactions: int = 0
        self._model: str = ""
        self._pending_tool_calls: dict = {}  # toolCallId -> {name, args_summary, start_time}
        self._consecutive_tool: str = ""
        self._consecutive_count: int = 0

    def find_active_session(self) -> Optional[Path]:
        """Find the most recently modified .jsonl session file."""
        try:
            jsonls = sorted(
                SESSIONS_DIR.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return jsonls[0] if jsonls else None
        except OSError:
            return None

    def poll(self):
        """Read new entries from session JSONL."""
        active = self.find_active_session()
        if not active:
            return

        # Session file changed
        if active != self._session_path:
            self._session_path = active
            self._session_pos = 0
            self._messages.clear()
            self._tool_history.clear()
            self._total_cost = 0.0
            self._cost_timestamps.clear()
            self._token_timestamps.clear()
            self._tool_counts.clear()
            self._tool_errors = 0
            self._tool_successes = 0
            self._compactions = 0
            self._pending_tool_calls.clear()

        try:
            with open(self._session_path, "r") as f:
                f.seek(0, 2)
                fsize = f.tell()
                if self._session_pos > fsize:
                    self._session_pos = 0
                f.seek(self._session_pos)
                data = f.read()
                self._session_pos = f.tell()
        except OSError:
            return

        if not data:
            return

        lines = data.split("\n")
        if data and not data.endswith("\n"):
            self._session_pos -= len(lines[-1].encode("utf-8", errors="replace"))
            lines = lines[:-1]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._process_session_entry(entry)

    def _summarize_args(self, tool_name: str, args: dict) -> str:
        """Create a short human-readable summary of tool arguments."""
        if not args:
            return ""
        if tool_name == "read":
            p = args.get("file_path") or args.get("path") or ""
            return Path(p).name if p else ""
        if tool_name == "write":
            p = args.get("file_path") or args.get("path") or ""
            return Path(p).name if p else ""
        if tool_name == "exec":
            cmd = args.get("command") or args.get("cmd") or ""
            if len(cmd) > 40:
                cmd = cmd[:37] + "..."
            return cmd
        if tool_name == "browser":
            action = args.get("action") or args.get("type") or ""
            url = args.get("url") or ""
            if url:
                # Just domain
                try:
                    from urllib.parse import urlparse
                    url = urlparse(url).netloc or url[:30]
                except Exception:
                    url = url[:30]
            return f"{action} {url}".strip()
        if tool_name == "web_search":
            return args.get("query", "")[:40]
        if tool_name in ("process", "session_status"):
            return args.get("action", "") or args.get("type", "")
        # Generic: first string value
        for v in args.values():
            if isinstance(v, str) and v:
                return v[:35]
        return ""

    def _process_session_entry(self, entry: dict):
        etype = entry.get("type", "")
        ts_str = entry.get("timestamp", "")
        ts_epoch = None
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_epoch = dt.timestamp()
            except (ValueError, AttributeError):
                ts_epoch = time.time()

        if etype == "model_change":
            self._model = entry.get("modelId", self._model)
            return

        if etype == "compaction":
            self._compactions += 1
            summary = entry.get("summary", "")
            if summary:
                self._last_compaction_summary = summary
            tokens_before = entry.get("tokensBefore", 0)
            if tokens_before:
                self._context_tokens = tokens_before // 2  # rough post-compaction estimate
            return

        if etype != "message":
            return

        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", [])

        # User message -> current task
        if role == "user":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block["text"]
                    # Strip timestamps/metadata prefixes
                    lines = text.strip().split("\n")
                    # Get the actual message content (skip [timestamp] lines)
                    clean_lines = []
                    for ln in lines:
                        if ln.startswith("[") and "]" in ln[:40]:
                            after = ln[ln.index("]") + 1:].strip()
                            if after:
                                clean_lines.append(after)
                        elif not ln.startswith("[message_id:") and not ln.startswith("[session:"):
                            clean_lines.append(ln)
                    if clean_lines:
                        self._last_user_msg = " ".join(clean_lines)[:120]
            return

        # Assistant message -> extract tool calls + cost
        if role == "assistant":
            usage = msg.get("usage", {})
            cost_info = usage.get("cost", {})
            cost = cost_info.get("total", 0) if isinstance(cost_info, dict) else 0
            if cost:
                self._total_cost += cost
                if ts_epoch:
                    self._cost_timestamps.append((ts_epoch, cost))

            total_tokens = usage.get("totalTokens", 0)
            if total_tokens and ts_epoch:
                self._token_timestamps.append((ts_epoch, total_tokens))
                # Use input tokens as proxy for context size
                inp = usage.get("input", 0) + usage.get("cacheRead", 0)
                if inp > self._context_tokens:
                    self._context_tokens = inp

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall":
                    tc_id = block.get("id", "")
                    tc_name = block.get("name", "unknown")
                    # Parse args
                    raw_args = block.get("input") or block.get("arguments") or {}
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {}
                    # Also check partialJson
                    if not raw_args:
                        pj = block.get("partialJson", "")
                        if pj:
                            try:
                                raw_args = json.loads(pj)
                            except json.JSONDecodeError:
                                raw_args = {}

                    args_summary = self._summarize_args(tc_name, raw_args)
                    self._pending_tool_calls[tc_id] = {
                        "name": tc_name,
                        "args": args_summary,
                        "start": ts_epoch or time.time(),
                    }

                    # Track consecutive same-tool calls
                    self._tool_counts[tc_name] = self._tool_counts.get(tc_name, 0) + 1
                    if tc_name == self._consecutive_tool:
                        self._consecutive_count += 1
                    else:
                        self._consecutive_tool = tc_name
                        self._consecutive_count = 1
            return

        # Tool result -> complete the tool call record
        if role == "toolResult":
            tc_id = msg.get("toolCallId", "")
            tc_name = msg.get("toolName", "")
            is_error = msg.get("isError", False)

            if is_error:
                self._tool_errors += 1
            else:
                self._tool_successes += 1

            pending = self._pending_tool_calls.pop(tc_id, None)
            duration = 0
            args_summary = ""
            if pending:
                duration = (ts_epoch or time.time()) - pending["start"]
                args_summary = pending["args"]
                tc_name = pending["name"] or tc_name

            # Extract result snippet
            result_snippet = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block["text"][:80]
                    result_snippet = t.replace("\n", " ")
                    break

            self._tool_history.append({
                "name": tc_name,
                "args": args_summary,
                "duration": duration,
                "ok": not is_error,
                "result": result_snippet,
                "time": ts_epoch or time.time(),
            })

    def get_insights(self, state: AgentState):
        """Fill the AgentState with session-derived insights."""
        # Current task
        if self._last_compaction_summary:
            # After compaction, the summary is the best description
            lines = self._last_compaction_summary.strip().split("\n")
            for ln in lines:
                ln = ln.strip()
                if ln and not ln.startswith("#") and not ln.startswith("-"):
                    state.current_task = ln[:100]
                    break
                elif ln.startswith("## ") or ln.startswith("# "):
                    state.current_task = ln.lstrip("# ")[:100]
                    break
            if not state.current_task and self._last_user_msg:
                state.current_task = self._last_user_msg
        elif self._last_user_msg:
            state.current_task = self._last_user_msg

        # Recent tools (last 3)
        state.recent_tools = list(self._tool_history)[-3:]

        # Cost
        state.total_cost = self._total_cost

        # Cost burn rate (last 5 min)
        now = time.time()
        cutoff = now - 300
        state.cost_last_5min = sum(c for t, c in self._cost_timestamps if t > cutoff)

        # Context tokens
        state.context_tokens = self._context_tokens
        state.model = self._model

        # Token burn rate
        if len(self._token_timestamps) >= 2:
            t0, tok0 = self._token_timestamps[0]
            t1, tok1 = self._token_timestamps[-1]
            dt = t1 - t0
            if dt > 30:
                state.tokens_per_min = (tok1 - tok0) / (dt / 60)

        # Tool distribution
        state.tool_counts = dict(self._tool_counts)
        state.tool_errors = self._tool_errors
        state.tool_successes = self._tool_successes
        state.compactions = self._compactions

        # Retry detection
        if self._consecutive_count >= 3:
            state.retry_detected = True
            state.retry_tool = self._consecutive_tool
        else:
            state.retry_detected = False

        # Warnings
        warnings = []
        total_calls = self._tool_errors + self._tool_successes
        if total_calls > 5 and self._tool_errors / total_calls > 0.3:
            warnings.append(f"High fail rate: {self._tool_errors}/{total_calls} tools errored")
        if state.retry_detected:
            warnings.append(f"Retry loop: {self._consecutive_tool} x{self._consecutive_count}")
        if self._context_tokens > 180000:
            warnings.append("Context near limit — compaction imminent")
        if self._compactions > 3:
            warnings.append(f"Heavy session: {self._compactions} compactions")
        # Cost warning
        if self._total_cost > 1.0:
            warnings.append(f"Session cost: ${self._total_cost:.2f}")
        state.warnings = warnings


# ─── RuntimePoller — tails /tmp/openclaw log for live status ─────────────────

class RuntimePoller:
    def __init__(self):
        self._log_path = self._current_log_path()
        self._log_date = self._log_path.name if self._log_path else None
        self._active_tools: dict = {}
        self._active_runs: dict = {}
        self._thinking_runs: set = set()
        self._cached_pid: Optional[int] = None
        self._last_pid_poll: float = 0
        self._file_pos: int = self._bootstrap_state()

    def _current_log_path(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return LOG_DIR / f"openclaw-{today}.log"

    def _bootstrap_state(self) -> int:
        if not self._log_path or not self._log_path.exists():
            return 0
        try:
            fsize = self._log_path.stat().st_size
        except OSError:
            return 0
        read_from = max(0, fsize - 131072)
        try:
            with open(self._log_path, "r") as f:
                f.seek(read_from)
                if read_from > 0:
                    f.readline()
                data = f.read()
        except OSError:
            return fsize

        runs_started = {}
        runs_done = set()
        tools_open = {}
        tools_closed = set()
        thinking = set()

        for line in data.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = entry.get("1", "")
            ts_str = entry.get("time", "")
            ts_epoch = None
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts_epoch = dt.timestamp()
                except (ValueError, AttributeError):
                    pass

            m = RE_RUN_START.search(msg)
            if m:
                runs_started[m.group(1)] = ts_epoch or time.time()
                runs_done.discard(m.group(1))
                thinking.discard(m.group(1))
                continue
            m = RE_RUN_DONE.search(msg)
            if m:
                runs_done.add(m.group(1))
                thinking.discard(m.group(1))
                continue
            m = RE_RUN_TIMEOUT.search(msg)
            if m:
                runs_done.add(m.group(1))
                continue
            m = RE_PROMPT_START.search(msg)
            if m:
                thinking.add(m.group(1))
                continue
            m = RE_PROMPT_END.search(msg)
            if m:
                thinking.discard(m.group(1))
                continue
            m = RE_TOOL_START.search(msg)
            if m:
                tc = re.search(r"toolCallId=(\S+)", msg)
                key = tc.group(1) if tc else f"{m.group(1)}:{m.group(2)}"
                tools_open[key] = m.group(2)
                tools_closed.discard(key)
                continue
            m = RE_TOOL_END.search(msg)
            if m:
                tc = re.search(r"toolCallId=(\S+)", msg)
                key = tc.group(1) if tc else f"{m.group(1)}:{m.group(2)}"
                tools_closed.add(key)
                continue

        active_rids = set(runs_started.keys()) - runs_done
        for rid in active_rids:
            self._active_runs[rid] = runs_started[rid]
            if rid in thinking:
                self._thinking_runs.add(rid)
        for key, tool in tools_open.items():
            if key not in tools_closed:
                self._active_tools[key] = tool

        return fsize

    def poll(self, state: AgentState):
        now = time.time()
        self._poll_log()
        if now - self._last_pid_poll > PID_POLL_INTERVAL:
            self._poll_pid()
            self._last_pid_poll = now

        # Derive status
        if state.gateway_pid is None and self._cached_pid is None:
            state.status = "offline"
            state.current_tool = None
            return
        state.gateway_pid = self._cached_pid

        if self._active_tools:
            state.status = "active"
            state.current_tool = list(self._active_tools.values())[-1]
            if self._active_runs:
                state.activity_start = min(self._active_runs.values())
        elif self._active_runs:
            state.status = "thinking"
            state.current_tool = None
            state.activity_start = min(self._active_runs.values())
        else:
            state.status = "idle"
            state.current_tool = None
            state.activity_start = None

        if state.status == "idle" and state.last_error_time and now - state.last_error_time < 300:
            state.status = "error"

    def _poll_log(self):
        expected = self._current_log_path()
        if self._log_date != expected.name:
            self._log_path = expected
            self._log_date = expected.name
            self._file_pos = 0
            self._active_tools.clear()
            self._active_runs.clear()
            self._thinking_runs.clear()

        if not self._log_path or not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r") as f:
                f.seek(0, 2)
                fsize = f.tell()
                if self._file_pos > fsize:
                    self._file_pos = 0
                f.seek(self._file_pos)
                data = f.read()
                self._file_pos = f.tell()
        except OSError:
            return

        if not data:
            return
        lines = data.split("\n")
        if data and not data.endswith("\n"):
            self._file_pos -= len(lines[-1].encode("utf-8", errors="replace"))
            lines = lines[:-1]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._process(entry)

    def _process(self, entry):
        msg = entry.get("1", "")
        level = entry.get("_meta", {}).get("logLevelName", "")

        m = RE_TOOL_START.search(msg)
        if m:
            tc = re.search(r"toolCallId=(\S+)", msg)
            key = tc.group(1) if tc else f"{m.group(1)}:{m.group(2)}"
            self._active_tools[key] = m.group(2)
            self._active_runs.setdefault(m.group(1), time.time())
            return
        m = RE_TOOL_END.search(msg)
        if m:
            tc = re.search(r"toolCallId=(\S+)", msg)
            key = tc.group(1) if tc else f"{m.group(1)}:{m.group(2)}"
            self._active_tools.pop(key, None)
            return
        m = RE_RUN_START.search(msg)
        if m:
            self._active_runs[m.group(1)] = time.time()
            return
        m = RE_RUN_DONE.search(msg)
        if m:
            self._active_runs.pop(m.group(1), None)
            self._thinking_runs.discard(m.group(1))
            self._active_tools.clear()
            return
        m = RE_PROMPT_START.search(msg)
        if m:
            self._thinking_runs.add(m.group(1))
            return
        m = RE_PROMPT_END.search(msg)
        if m:
            self._thinking_runs.discard(m.group(1))
            return
        m = RE_RUN_TIMEOUT.search(msg)
        if m:
            self._active_runs.pop(m.group(1), None)
            self._thinking_runs.discard(m.group(1))
            self._active_tools.clear()
            return

    def _poll_pid(self):
        try:
            result = subprocess.run(
                ["pgrep", "-f", "openclaw-gateway"],
                capture_output=True, text=True, timeout=5,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            if pids:
                new_pid = int(pids[0])
                if self._cached_pid and new_pid != self._cached_pid:
                    self._active_tools.clear()
                    self._active_runs.clear()
                self._cached_pid = new_pid
            else:
                self._cached_pid = None
        except (subprocess.SubprocessError, ValueError):
            self._cached_pid = None


# ─── Drawing helpers (standalone to avoid PyObjC method conflicts) ────────────

def _dt(view, ctx, text, x, y, size, bold=False, color=COLOR_WHITE):
    """Draw text at (x, y)."""
    attrs = {
        AppKit.NSFontAttributeName: (
            AppKit.NSFont.boldSystemFontOfSize_(size) if bold
            else AppKit.NSFont.monospacedSystemFontOfSize_weight_(size, AppKit.NSFontWeightRegular)
        ),
        AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithRed_green_blue_alpha_(*color),
    }
    ns_str = Foundation.NSAttributedString.alloc().initWithString_attributes_(text, attrs)
    ns_str.drawAtPoint_(Foundation.NSMakePoint(x, y))


def _ds(ctx, y, w):
    """Draw separator line."""
    Quartz.CGContextSetRGBStrokeColor(ctx, *COLOR_SEPARATOR)
    Quartz.CGContextSetLineWidth(ctx, 0.5)
    Quartz.CGContextMoveToPoint(ctx, 10, y)
    Quartz.CGContextAddLineToPoint(ctx, w - 10, y)
    Quartz.CGContextStrokePath(ctx)


# ─── OverlayView — renders the intelligence dashboard ────────────────────────

class OverlayView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(OverlayView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._state = AgentState()
        return self

    def setState_(self, state: AgentState):
        self._state = state
        self.setNeedsDisplay_(True)

    def isFlipped(self):
        return True

    def drawRect_(self, rect):
        ctx = AppKit.NSGraphicsContext.currentContext().CGContext()
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height

        # Background
        Quartz.CGContextSetRGBFillColor(ctx, *BG_COLOR)
        path = Quartz.CGPathCreateWithRoundedRect(
            Quartz.CGRectMake(0, 0, w, h), 10, 10, None
        )
        Quartz.CGContextAddPath(ctx, path)
        Quartz.CGContextFillPath(ctx)

        s = self._state
        y = 10

        # ── Row 1: [dot] STATUS  $cost  goal text ──
        dot_color = {
            "active": COLOR_GREEN, "thinking": COLOR_YELLOW,
            "idle": COLOR_GRAY, "error": COLOR_RED, "offline": COLOR_DIM,
        }.get(s.status, COLOR_DIM)

        dot_x, dot_y, dot_r = 14, y + 5, 5
        Quartz.CGContextSetRGBFillColor(ctx, *dot_color)
        Quartz.CGContextFillEllipseInRect(
            ctx, Quartz.CGRectMake(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2)
        )

        label = s.status.upper()
        _dt(self, ctx, label, 24, y, 10, bold=True, color=dot_color)
        # Position after status label
        x_after = 24 + len(label) * 7 + 4

        if s.current_tool:
            _dt(self, ctx, s.current_tool, x_after, y, 10, color=COLOR_WHITE)
            if s.activity_start:
                elapsed = time.time() - s.activity_start
                mn, sec = divmod(int(elapsed), 60)
                _dt(self, ctx, f"{mn}:{sec:02d}", w - 40, y, 10, color=COLOR_MUTED)
        else:
            # Cost + goal on the status line
            cost_str = f"${s.total_cost:.2f}"
            _dt(self, ctx, cost_str, x_after, y, 10, color=COLOR_MUTED)
            goal_x = x_after + len(cost_str) * 7 + 6
            if s.current_task:
                task = s.current_task
                max_chars = int((w - goal_x - 6) / 6)
                if max_chars > 3:
                    if len(task) > max_chars:
                        task = task[:max_chars - 3] + "..."
                    _dt(self, ctx, task, goal_x, y, 10, color=COLOR_CYAN)
        y += 18

        # ── Row 2: ctx | err | comp | tools — all on one line ──
        ctx_k = s.context_tokens // 1000 if s.context_tokens else 0
        parts = [f"{ctx_k}k"]
        total = s.tool_errors + s.tool_successes
        if total > 0:
            pct = int(s.tool_errors / total * 100)
            parts.append(f"e:{pct}%")
        if s.compactions > 0:
            parts.append(f"c:{s.compactions}")
        # Add top tools inline
        if s.tool_counts:
            sorted_tools = sorted(s.tool_counts.items(), key=lambda x: -x[1])[:4]
            for tname, cnt in sorted_tools:
                parts.append(f"{tname}:{cnt}")
        _dt(self, ctx, "  ".join(parts), 14, y, 9, color=COLOR_MUTED)
        y += 16

        _ds(ctx, y, w)
        y += 6

        # ── Recent activity (3 items) ──
        _dt(self, ctx, "ACTIVITY", 14, y, 9, bold=True, color=COLOR_MUTED)
        y += 14

        if s.recent_tools:
            for tool in s.recent_tools[-3:]:
                name = tool["name"]
                args = tool.get("args", "")
                dur = tool.get("duration", 0)
                ok = tool.get("ok", True)

                indicator_color = COLOR_GREEN if ok else COLOR_RED
                _dt(self, ctx, "+" if ok else "x", 14, y, 9, bold=True, color=indicator_color)
                _dt(self, ctx, name, 26, y, 9, bold=True, color=COLOR_WHITE)

                if args:
                    if len(args) > 26:
                        args = args[:23] + "..."
                    _dt(self, ctx, args, 95, y, 9, color=COLOR_MUTED)

                if dur > 0:
                    dur_str = f"{dur:.1f}s" if dur < 60 else f"{dur/60:.1f}m"
                    _dt(self, ctx, dur_str, w - 40, y, 9, color=COLOR_MUTED)

                y += 14
        else:
            _dt(self, ctx, "(waiting for activity)", 14, y, 9, color=COLOR_DIM)
            y += 14

        # ── Warnings (max 3) ──
        if s.warnings:
            _ds(ctx, y + 2, w)
            y += 8
            for warn in s.warnings[:3]:
                Quartz.CGContextSetRGBFillColor(ctx, *COLOR_ERROR_BG)
                Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(6, y - 3, w - 12, 16))
                if len(warn) > 45:
                    warn = warn[:42] + "..."
                _dt(self, ctx, f"! {warn}", 10, y, 9, color=COLOR_ORANGE)
                y += 18

        # ── Resize window to fit content ──
        needed_h = y + 8
        if abs(h - needed_h) > 4:
            panel = self.window()
            if panel:
                frame = panel.frame()
                delta = needed_h - h
                new_frame = Foundation.NSMakeRect(
                    frame.origin.x,
                    frame.origin.y - delta,
                    frame.size.width,
                    needed_h,
                )
                panel.setFrame_display_(new_frame, True)
                self.setFrameSize_(Foundation.NSMakeSize(w, needed_h))


# ─── OverlayWindow ───────────────────────────────────────────────────────────

def create_overlay_window() -> AppKit.NSPanel:
    screen = AppKit.NSScreen.mainScreen()
    vf = screen.visibleFrame()
    x = vf.origin.x + vf.size.width - WIN_W - MARGIN
    y = vf.origin.y + MARGIN

    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        Foundation.NSMakeRect(x, y, WIN_W, WIN_H),
        style, AppKit.NSBackingStoreBuffered, False,
    )
    panel.setLevel_(AppKit.NSStatusWindowLevel)
    panel.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setHasShadow_(True)
    panel.setIgnoresMouseEvents_(True)
    panel.setFloatingPanel_(True)
    panel.setHidesOnDeactivate_(False)

    view = OverlayView.alloc().initWithFrame_(
        Foundation.NSMakeRect(0, 0, WIN_W, WIN_H)
    )
    panel.setContentView_(view)
    panel.orderFrontRegardless()
    return panel


# ─── App Delegate ────────────────────────────────────────────────────────────

class OpenClawOverlayApp(AppKit.NSObject):
    def init(self):
        self = objc.super(OpenClawOverlayApp, self).init()
        if self is None:
            return None
        self._runtime = RuntimePoller()
        self._session = SessionAnalyzer()
        self._panel = None
        self._view = None
        self._timer = None
        self._session_timer = None
        return self

    def applicationDidFinishLaunching_(self, notification):
        AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        self._panel = create_overlay_window()
        self._view = self._panel.contentView()

        # Initial polls
        self._runtime._poll_pid()
        self._runtime._last_pid_poll = time.time()
        self.doFullPoll()

        # Fast timer for runtime status (2s)
        self._timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            POLL_INTERVAL, self, "tick:", None, True,
        )
        Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._timer, Foundation.NSRunLoopCommonModes,
        )

        # Slower timer for session analysis (5s)
        self._session_timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            SESSION_POLL_INTERVAL, self, "sessionTick:", None, True,
        )
        Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._session_timer, Foundation.NSRunLoopCommonModes,
        )

        # Screen change
        Foundation.NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self, "screenChanged:",
            AppKit.NSApplicationDidChangeScreenParametersNotification, None,
        )

        # Option-drag support
        def handle_flags(event):
            opt = bool(event.modifierFlags() & AppKit.NSEventModifierFlagOption)
            self._panel.setIgnoresMouseEvents_(not opt)
            self._panel.setMovableByWindowBackground_(opt)

        def handle_flags_local(event):
            handle_flags(event)
            return event

        AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            AppKit.NSEventMaskFlagsChanged, handle_flags,
        )
        AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            AppKit.NSEventMaskFlagsChanged, handle_flags_local,
        )

        print("OpenClaw intelligence overlay running. Ctrl+C to exit.", flush=True)

    def doFullPoll(self):
        state = AgentState()
        self._runtime.poll(state)
        self._session.poll()
        self._session.get_insights(state)
        self._view.setState_(state)

    def tick_(self, timer):
        state = AgentState()
        self._runtime.poll(state)
        self._session.get_insights(state)
        self._view.setState_(state)

    def sessionTick_(self, timer):
        self._session.poll()

    def screenChanged_(self, notification):
        screen = AppKit.NSScreen.mainScreen()
        if not screen:
            return
        vf = screen.visibleFrame()
        x = vf.origin.x + vf.size.width - WIN_W - MARGIN
        y = vf.origin.y + MARGIN
        self._panel.setFrameOrigin_(Foundation.NSMakePoint(x, y))

    def applicationWillTerminate_(self, notification):
        if self._timer:
            self._timer.invalidate()
        if self._session_timer:
            self._session_timer.invalidate()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    app = AppKit.NSApplication.sharedApplication()
    delegate = OpenClawOverlayApp.alloc().init()
    app.setDelegate_(delegate)

    signal.signal(signal.SIGINT, lambda *_: AppKit.NSApp.terminate_(None))
    signal.signal(signal.SIGTERM, lambda *_: AppKit.NSApp.terminate_(None))

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
