# openclaw-overlay

A floating macOS status overlay for [OpenClaw](https://openclaw.dev) that provides **real-time intelligence** about your agent's activity — things the built-in TUI doesn't show.

![macOS](https://img.shields.io/badge/platform-macOS-lightgrey) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![PyObjC](https://img.shields.io/badge/requires-PyObjC-orange)

## What It Shows (That the TUI Doesn't)

| Feature | Description |
|---------|-------------|
| **Session cost** | Running `$` total + 5-minute burn rate |
| **Activity feed** | Last 5 tool calls with arguments, duration, and pass/fail |
| **Tool distribution** | Which tools are used most (e.g. `exec:12 read:8 browser:5`) |
| **Error rate** | Percentage of tool calls that failed |
| **Retry loop detection** | Warns when the agent calls the same tool 3+ times consecutively |
| **Context tracking** | Token count + compaction count |
| **Current task** | Extracted from last user message or compaction summary |
| **Live status** | Active/Thinking/Idle/Error/Offline with elapsed timer |

The overlay parses the **session JSONL transcript** (full conversation with tool arguments, results, and cost data) — not just the runtime status log.

## Screenshot

```
+--------------------------------------------------------+
|  * ACTIVE   exec                              1:42     |
|  Fill out the fillable PDF form...                     |
|  ---------------------------------------------------   |
|  Cost: $0.847  Ctx: 182k  err:12%                     |
|  Burn: $0.092/5m                                       |
|  ---------------------------------------------------   |
|  RECENT ACTIVITY                                       |
|  + exec    python3 fill_pdf.py              3.2s       |
|  + read    template.pdf                     0.1s       |
|  x browser click submit-btn                 7.8s       |
|  + write   output.pdf                       0.4s       |
|  + exec    ls -la /tmp/output               0.2s       |
|  ---------------------------------------------------   |
|  TOOLS                                                 |
|  exec:12  read:8  write:4  browser:3                   |
|  ---------------------------------------------------   |
|  ! High fail rate: 6/18 tools errored                  |
|  ! Session cost: $0.85                                 |
+--------------------------------------------------------+
```

## Requirements

- **macOS** (uses native AppKit/Quartz via PyObjC)
- **Python 3.10+**
- **PyObjC** (`pip install pyobjc`)
- **OpenClaw** running with gateway active

## Install

```bash
# Clone
git clone https://github.com/shivshil/openclaw-overlay.git
cd openclaw-overlay

# Install dependency
pip install pyobjc

# Run
python3 openclaw_overlay.py
```

Or just download the single file:

```bash
curl -O https://raw.githubusercontent.com/shivshil/openclaw-overlay/main/openclaw_overlay.py
pip install pyobjc
python3 openclaw_overlay.py
```

## Usage

```bash
python3 openclaw_overlay.py
```

The overlay appears in the **bottom-right corner** of your screen.

- **Always on top** across all Spaces/desktops
- **Click-through** by default — it won't interfere with your work
- **Hold Option** to drag and reposition
- **Ctrl+C** to quit

## How It Works

The overlay reads from two data sources:

1. **Runtime log** (`/tmp/openclaw/openclaw-YYYY-MM-DD.log`) — JSONL with tool start/end lifecycle events. Polled every 2 seconds for live status.

2. **Session JSONL** (`~/.openclaw/agents/main/sessions/*.jsonl`) — Full conversation transcript including tool call arguments, results, token usage, and cost breakdowns. Polled every 5 seconds for analytics.

### Status Detection

| Status | Condition |
|--------|-----------|
| **ACTIVE** (green) | Tool currently executing |
| **THINKING** (yellow) | Run open, no active tool (model is generating) |
| **IDLE** (gray) | No active runs |
| **ERROR** (red) | Idle + recent error within 5 minutes |
| **OFFLINE** (dim) | Gateway process not found |

### Warnings

The overlay generates warnings when it detects:

- **High error rate** — more than 30% of tool calls failing
- **Retry loops** — same tool called 3+ times consecutively (agent may be stuck)
- **Context near limit** — approaching 200k token compaction threshold
- **Heavy session** — more than 3 compactions in one session
- **High cost** — session spend exceeds $1.00

## Configuration

Edit the constants at the top of `openclaw_overlay.py`:

```python
POLL_INTERVAL = 2.0          # Runtime log poll frequency (seconds)
SESSION_POLL_INTERVAL = 5.0  # Session JSONL poll frequency (seconds)
WIN_W, WIN_H = 380, 310      # Window dimensions
MARGIN = 12                   # Distance from screen edge
```

## License

MIT
