# OpenEnv Disaster Relief Coordination: Meta-RL Benchmark Portfolio
> **Purpose**: A comprehensive, zero-information-loss technical documentation of the **Disaster Relief Coordination** OpenEnv benchmark repository ([Meta-RL-hack](file:///Users/varsham/Desktop/ujwal/Meta-RL-hack)). Tailored for resume updates, technical portfolio showcases, AI research engineering interviews, and multi-agent RL systems evaluation.

---

## Executive Summary & Technical Skill Matrix

### Project Overview
**Disaster Relief Coordination** is an **OpenEnv-compatible** reinforcement learning environment and benchmark designed for evaluating and training LLM agents and multi-agent systems in complex, multi-objective decision-making under uncertainty. The agent acts as an emergency operations center coordinator managing concurrent triage, resource allocation, terrain physics constraints, false alarm detection, and operation monitoring under strict time deadlines.

### Specialized Technical Skill Matrix
- **RL Environment & Benchmark Design**: OpenEnv Framework, Gymnasium-style `reset()` / `step()` / `grade()` lifecycle, Deterministic Seeded Scenario Generators, Immutable State Isolation.
- **Multi-Agent Systems & Tool Delegation**: Hierarchical LLM Coordinator pattern, Tool-Mediated Delegation (`call_intake_agent`, `call_dispatch_agent`, `call_monitor_agent`), Structured JSON Tool Calling (`{"tool": "...", "args": {...}}`).
- **Reward Engineering & Multi-Axis Graders**: Per-step dense reward shaping, Temporal urgency multipliers, Exploit-resistant counterfactual grading (evaluating idle matching resources during critical expirations), False alarm F1-score optimization.
- **Simulation Mechanics & Physics Modeling**: Terrain accessibility modeling (flood depth levels $\ge 2$ requiring specialized amphibious/aerial units), Dynamic road blockages, Comms blackouts, Mid-episode event arrival streams.
- **Serving & System Architecture**: FastAPI REST API, Docker non-root deployment, OpenAI-Compatible LLM Client (`inference.py`), Deterministic Flood-Aware & Type-Matched Heuristic Baseline Engine.

---

## 1. Resume-Ready Experience Bullet Points

### OpenEnv Benchmark & RL Environment Engineering
* **Architected an OpenEnv-Compatible Multi-Agent Benchmark for Disaster Relief Coordination**, designing a deterministic simulation environment (`DisasterReliefEnv`) to evaluate LLM agents on concurrent triage, resource allocation, and dynamic replanning.
* **Engineered a Counterfactual Multi-Axis Grader**, evaluating agent performance across 6 non-exploitable dimensions (Resolution Rate, Critical Deadline Adherence, False Alarm F1-Score, Resource Capability Alignment, Monitoring Quality, and Counterfactual Idle-Resource Penalties).
* **Formulated a Dense Per-Step Reward Shaping Function**, incorporating temporal urgency decay multipliers ($2.0\times \rightarrow 1.0\times$), penalty structures for misassigned resource capabilities, and anti-gaming penalties for repeated or malformed actions.

### Multi-Agent Systems & LLM Tool Orchestration
* **Designed a Hierarchical Multi-Agent Delegation Framework**, featuring a centralized LLM Coordinator delegating domain-specific workflows to specialized sub-agents (`Intake`, `Dispatch`, `Monitor`) via a 12-tool JSON API.
* **Built an Adaptive Context-Optimized Prompting Pipeline**, compressing multi-zone world states, flood depth flags, fleet availability, and active assignments into a $<800$-token prompt summary for efficient real-time LLM inference.
* **Developed a Flood-Aware & Resource-Matched Heuristic Baseline**, establishing a deterministic decision engine achieving benchmark calibration scores of $0.46$ (Easy), $0.79$ (Medium), and $0.48$ (Hard) without external model dependencies.

### Serving Infrastructure & DevOps
* **Containerized and Deployed the OpenEnv Server via Docker & FastAPI**, implementing non-root execution, asynchronous HTTP endpoints (`/reset`, `/step`, `/grade`, `/state`), and standardized benchmark STDOUT logging wrappers (`[START]`, `[STEP]`, `[END]`).

---

## 2. Environment Architecture & System Design

```
===================================================================================
DISASTER RELIEF COORDINATION OPENENV BENCHMARK ARCHITECTURE
===================================================================================
                               ┌──────────────────────────┐
                               │     LLM COORDINATOR      │
                               │   (Decision Maker Agent) │
                               └────────────┬─────────────┘
                                            │ Structured JSON Tool Call
                                            ▼
                               ┌──────────────────────────┐
                               │  DisasterReliefEnv API   │
                               │  (reset / step / grade)  │
                               └────────────┬─────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
   ┌───────────────────────┐   ┌───────────────────────┐    ┌───────────────────────┐
   │     INTAKE AGENT      │   │    DISPATCH AGENT     │    │     MONITOR AGENT     │
   │• classify_report      │   │• get_resources        │    │• check_operation      │
   │• assess_report_urgency│   │• send_resource        │    │• close_case           │
   │• verify_report        │   │• reroute_resource     │    │• mark_false_report    │
   └───────────────────────┘   └───────────────────────┘    └───────────────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │    WorldState Physics    │
                               │• advance_time() clock    │
                               │• Flood depth & road block│
                               │• Comms blackout deltas   │
                               └──────────────────────────┘
===================================================================================
```

### 2.1 The WorldState Simulation Engine (`src/env/state.py`)
The `WorldState` class maintains the complete simulation state across discrete step ticks:
- **Time Clock (`advance_time()`)**: Executes simulation logic at each step:
  - Decrements assignment travel ETAs and moves status from `EN_ROUTE` $\rightarrow$ `ON_SITE` $\rightarrow$ `COMPLETED`.
  - Progresses report deadlines and triggers `EXPIRED` status for unresolved incidents.
  - Clears access blockages and restores communications blackouts mid-episode.
  - Releases resources back to `AVAILABLE` status upon completion.
  - Logs resource availability snapshots into `availability_log` for post-hoc counterfactual grading.

### 2.2 Domain Object Taxonomy (`src/env/models.py`)
- **Report**: Captures incident metadata (`category`, `urgency` [0-10], `zone_id`, `is_critical`, `deadline_step`, `status`, `verified`, `verification_confidence`, `ground_truth`).
- **Resource**: Captures fleet capability (`type`, `status`, `zone_id`, `can_traverse_flood`).
- **Zone**: Defines regional environmental constraints (`flood_depth_level` [0-3], `access_blocked`, `comms_blackout`, `severity`).
- **Assignment**: Tracks dispatch operations (`resource_id`, `report_id`, `travel_steps`, `status`, `stuck`).

---

## 3. Tool API Space (12 Tool Definitions)

Actions are supplied as JSON objects: `{"tool": "<name>", "args": {<key_value_pairs>}}`.

| Tool Name | Type | Key Arguments | Functional Purpose |
| :--- | :--- | :--- | :--- |
| `call_intake_agent` | **Delegation** | `report_id` | Executes full 3-step triage pipeline (`classify` + `urgency` + `verify`) in **1 step**. |
| `call_dispatch_agent` | **Delegation** | `resource_id, report_id` | Inspects inventory status then dispatches resource to report. |
| `call_monitor_agent` | **Delegation** | `target_id, instruction` | Monitors assignment status; auto-closes if resolved. |
| `classify_report` | Direct | `report_id` | Determines incident category (`flood`, `medical`, `fire`, etc.). |
| `assess_report_urgency` | Direct | `report_id` | Evaluates urgency score on a scale of $0\text{--}10$. |
| `verify_report` | Direct | `report_id` | Authenticates authenticity (reveals confidence score $\in [0, 1]$). |
| `send_resource` | Direct | `resource_id, report_id` | Dispatches specified resource to incident location. |
| `get_resources` | Direct | *(none)* | Returns full fleet status and availability. |
| `reroute_resource` | Direct | `resource_id` | Reroutes a resource stuck on a blocked access route. |
| `check_operation` | Direct | `target_id` | Queries status of a report or active assignment. |
| `close_case` | Direct | `report_id` | Closes completed operation and frees resource. |
| `mark_false_report` | Direct | `report_id, reason` | Flags report as false alarm/duplicate (confidence $< 0.3$). |

---

## 4. Benchmark Task Hierarchy & Score Calibration

### 4.1 Task Progression Ladder

| Task Identifier | Difficulty | Max Steps | Zones | Total Reports | Critical Reports | False/Dup Reports | Fleet Size | Active Grader Axes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `task1_flood_easy` | Easy | 12 | 1 | 6 | 2 | 1 | 5 | 3 Axes |
| `task2_storm_medium` | Medium | 20 | 3 | 11 | 3 | 3 | 6 | 5 Axes |
| `task3_cascade_hard` | Hard | 30 | 5 | 17 | 4 | 5 | 6 | 6 Axes |

### 4.2 Resource Capabilities & Terrain Requirements

| Resource Type | Flood Traversable? (`flood_depth >= 2`) | Preferred Incident Match |
| :--- | :---: | :--- |
| **Rescue Boat** | **Yes** | Flood rescues & water evacuations |
| **Helicopter** | **Yes** | All-terrain rapid response & critical transport |
| **Ambulance** | No | Medical transport (shallow/dry ground) |
| **Medical Team** | No | On-site emergency triage & treatment |
| **Supply Truck** | No | Mass evacuation & cargo logistics |
| **Engineering Crew** | No | Structural collapse, fire, road clearance |

*Constraint Rule*: Sending a non-flood resource (e.g., Ambulance) to a zone with `flood_depth >= 2` results in silent dispatch failure, wasting time and resources.

### 4.3 Benchmark Calibration Matrix

| Agent Strategy | `task1_flood_easy` | `task2_storm_medium` | `task3_cascade_hard` |
| :--- | :---: | :---: | :---: |
| **Do-Nothing / Random** | ~0.00 | ~0.00 | ~0.00 |
| **Heuristic Baseline** | **0.46** | **0.79** | **0.48** |
| **Perfect-Knowledge Oracle** | **0.71** | **0.79** | **0.50** |
| **Theoretical Maximum** | 1.00 | 1.00 | 1.00 |

*Insight*: The Hard task has an oracle score ceiling of $\sim 0.50$ due to genuine structural difficulty (severe resource scarcity, extreme flood depth, late-arriving reports), providing significant evaluation headroom for frontier reasoning models.

---

## 5. Reward Design & Counterfactual Grader Mechanics

### 5.1 Step-Level Dense Reward Formulation (`src/env/rewards.py`)
Per-step reward $R_{\text{step}}$ provides dense feedback:

$$R_{\text{step}} = R_{\text{base}} + R_{\text{triage}} + R_{\text{dispatch}} + R_{\text{flag}} + P_{\text{repeat}} + P_{\text{error}}$$

- **Base Survival**: $R_{\text{base}} = +0.1$
- **Delegated Triage**: $+1.0$ (classify), $+1.0$ (urgency), $+0.5$ (verify via `call_intake_agent`)
- **Resource Dispatch**:
  - Correct resource type: $+2.0 \times \text{UrgencyMultiplier}$
  - Incorrect resource type: $-1.0$
  - Dispatching to false alarm: $-1.0$
- **False Alarm Flagging**:
  - Correct flag (confidence $< 0.3$): $+1.0$
  - False positive flag (flagging real report): $-1.5$
  - Flagging critical real report: $-2.0$
- **Temporal Urgency Multiplier**: $\text{Mult} = 1.0 + \frac{\text{Deadline} - \text{Step}}{\text{MaxSteps}} \in [1.0, 2.0]$
- **Penalties**: $-0.5$ for repeating the exact same action key or sending malformed arguments.

### 5.2 Multi-Axis Episode Graders (`src/env/graders.py`)

Final episode score $S \in [0.0, 1.0]$ is computed deterministically at episode end:

#### Task 1 Score Formulation
$$S_{\text{task1}} = 0.40 \cdot S_{\text{resolution}} + 0.30 \cdot S_{\text{critical}} + 0.30 \cdot S_{\text{efficiency}}$$

#### Task 2 Score Formulation
$$S_{\text{task2}} = 0.30 \cdot S_{\text{resolution}} + 0.25 \cdot S_{\text{critical}} + 0.15 \cdot F1_{\text{false\_alarm}} + 0.15 \cdot S_{\text{resource\_match}} + 0.15 \cdot S_{\text{counterfactual}}$$

#### Task 3 Score Formulation
$$S_{\text{task3}} = 0.30 \cdot S_{\text{resolution}} + 0.25 \cdot S_{\text{critical}} + 0.15 \cdot F1_{\text{false\_alarm}} + 0.10 \cdot S_{\text{resource\_match}} + 0.10 \cdot S_{\text{monitoring}} + 0.10 \cdot S_{\text{counterfactual}}$$

### 5.3 Grader Mathematical Sub-Routines

1. **Resolution Score ($S_{\text{resolution}}$)**:
   $$S_{\text{resolution}} = \frac{|\text{Resolved Real Reports}|}{|\text{Total Real Reports}|}$$

2. **Critical Deadline Score ($S_{\text{critical}}$)**:
   $$S_{\text{critical}} = \frac{\sum_{r \in \text{Critical}} \mathbb{I}(\text{Resolved Before Deadline}) + 0.5 \cdot \mathbb{I}(\text{Resolved Late})}{|\text{Total Critical Reports}|}$$

3. **False Alarm F1-Score ($F1_{\text{false\_alarm}}$)**:
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

4. **Counterfactual Score ($S_{\text{counterfactual}}$)**:
   $$S_{\text{counterfactual}} = 1.0 - \frac{|\text{Preventable Expired Critical Reports}|}{|\text{Total Expired Critical Reports}|}$$
   *Preventable Rule*: A critical report expiration is deemed *preventable* if a matching resource was logged as `AVAILABLE` in `availability_log` at any step between report creation and deadline.

---

## 6. HTTP API Specification & Deployment

### Server Endpoints (`app.py` / Port 7860)

| Endpoint | Method | Input Payload | Output Payload / Response |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | *(none)* | `{"status": "ok", "benchmark": "disaster-relief-coordination"}` |
| `/tasks` | `GET` | *(none)* | `{"tasks": ["task1_flood_easy", "task2_storm_medium", "task3_cascade_hard"]}` |
| `/reset` | `POST` | `{"task_name": "task1_flood_easy", "seed": 42}` | `StepResult` JSON |
| `/step` | `POST` | `{"tool": "call_intake_agent", "args": {"report_id": "RPT-001"}}` | `StepResult` JSON (`observation`, `reward`, `done`, `info`) |
| `/grade` | `POST` | *(none)* | `GradeResult` JSON (`score`, `breakdown`) |
| `/state` | `GET` | *(none)* | Internal `WorldState` snapshot |

### Deployment Specifications
- **Base Image**: `python:3.11-slim`
- **Security**: Non-root user execution (`useradd -m -u 1000 user`)
- **Package Management**: `uv` fast dependency installer / `pyproject.toml`
- **Execution Command**: `uvicorn app:app --host 0.0.0.0 --port 7860`

---
*Document automatically generated from code inspection of [Meta-RL-hack](file:///Users/varsham/Desktop/ujwal/Meta-RL-hack).*
