# Kennel: Model Lifecycle Management

> Manage AI models like a kennel manages working dogs — fielded, kenneled, retired, with duty rosters for cost-managed shift rotation.

Part of the **Working Animal Architecture** from [SuperInstance](https://github.com/SuperInstance).

## Why?

Models are working animals. You don't leave a sheepdog on duty 24/7 — it gets tired, sloppy, and expensive. The same is true of LLM instances in production:

- **Overgrazing** — a single model handling too much context degrades output quality (context poisoning).
- **Cost burn** — always-on premium models drain budgets silently.
- **No rotation discipline** — ad-hoc switching with no records makes incident post-mortoms painful.

Kennel fixes this by treating each model instance as a **working animal** with shifts, rest periods, and a retirement path.

## Model States

```
 ┌─────────┐    deploy()   ┌──────────┐    retire()   ┌─────────┐
 │  POOLED  │ ───────────► │ FIELDLED  │ ───────────► │ RETIRED  │
 │ (waiting)│              │ (working) │              │ (archived)│
 └─────────┘               └──────────┘               └─────────┘
                                 │
                          kennel()
                                 ▼
                           ┌──────────┐
                           │KENNELED   │
                           │ (resting) │
                           └──────────┘
                                 │
                          deploy()
                                 ▼
                           ┌──────────┐
                           │ FIELDLED  │
                           └──────────┘
```

| State | Meaning |
|---|---|
| `POOLED` | Available but not yet deployed. Waiting in the pool. |
| `FIELDLED` | Actively working. Serving requests. |
| `KENNELED` | Resting between shifts. Not serving traffic. |
| `RETIRED` | Archived. No longer in active rotation. Historical records kept. |

## Quick Start

```bash
pip install kennel-manager
```

```python
from kennel import Kennel

kennel = Kennel()

# Deploy a model to active duty
model = kennel.deploy(
    name="gpt-4o",
    alias="scout",
    provider="openai",
    max_shift_hours=6,
)

# Check who's on duty
print(kennel.roster())
# [{ alias: "scout", model: "gpt-4o", state: "FIELDLED", shift_started: ... }]

# Kennel a model for rest
kennel.kennel("scout")

# Retire a model permanently
kennel.retire("scout")

# Full status report
print(kennel.status())
```

## Duty Rosters

Each model gets a **DutyRoster** — a shift schedule that prevents overgrazing:

```python
roster = kennel.roster()
for entry in roster:
    print(f"{entry['alias']}: {entry['state']} — {entry['shift_hours']:.1f}h on shift")
```

Rosters enforce:

- **Max shift length** — auto-kennel after N hours
- **Min rest period** — must rest before re-deployment
- **Rotation policy** — round-robin or weighted selection from the pool
- **Cost ceiling** — auto-kennel when shift cost exceeds budget

## Cost Tracking

Every shift produces a `ShiftRecord` with token counts and cost:

```python
for record in kennel.cost_history("scout"):
    print(f"  {record.started_at}: ${record.cost_usd:.4f} ({record.input_tokens} in / {record.output_tokens} out)")
```

Aggregate costs per model, per day, or per roster cycle:

```python
total = kennel.cost_summary()
for model, cost in total.items():
    print(f"  {model}: ${cost:.2f}")
```

## Architecture

```
kennel/
├── __init__.py     # Kennel — main API class
├── model.py        # ModelInstance, DutyRoster, ShiftRecord dataclasses
├── rotation.py     # Shift rotation & anti-overgrazing logic
└── cost.py         # Per-shift cost tracking & aggregation
```

## Working Animal Architecture

Kennel is one animal in the architecture:

| Component | Role | Repo |
|---|---|---|
| **Kennel** | Model lifecycle & shift management | This repo |
| Pack | Multi-model orchestration | SuperInstance/pack |
| Handler | Request routing & prompt engineering | SuperInstance/handler |
| Veterinarian | Model health checks & quality evals | SuperInstance/vet |

## License

MIT
