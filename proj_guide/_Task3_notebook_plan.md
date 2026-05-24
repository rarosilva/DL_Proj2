# Task 3 Notebook Plan - Simple Notebook-First Version

This plan is intentionally simple.

Decision for now:
- implement everything directly in the notebook
- reuse only space_race_env.py from project files
- do not extract code to src at this stage
- keep the same style and flow used in task2.ipynb

Main objective:
- continue from the best Task 2 pipeline and improve it through exploration strategy experiments for Task 3

---
# **0.1 Critical Review of Notes (Validated Decisions)**

This section captures which notes are valid and what we changed in the plan.

Valid and adopted:
- use fixed training steps (or fixed env interactions) as the primary fairness budget
- report wall-clock time as secondary efficiency metric, not as the main comparison axis
- keep Task 2 winning backbone fixed (architecture, preprocess, frame stacking, replay, target settings)
- implement Boltzmann with temperature-scaled softmax probabilities
- evaluate with deterministic policy (no exploration) for both strategies
- compare strategies with multiple metrics, not only mean reward

Valid but adjusted:
- "5 seeds for final comparison": good target, but use 3 seeds during tuning and 5 for final if runtime budget allows
- "statistical significance": avoid overclaiming; report mean/std and confidence intervals (bootstrap or t-interval) and keep interpretation conservative

Rejected or corrected:
- arbitrary convergence threshold (example: score > 8.0) is not robust; use relative convergence metrics instead
- wording referencing "Snake" is incorrect for this project; all analysis must reference SpaceRace

---
# **0.2 Non-Negotiable Protocol Rules**

These are mandatory in this notebook so comparison stays valid.

- same backbone for all runs (from Task 2 best config)
- same seed list for epsilon and boltzmann grids
- same training-step budget per run
- same evaluation protocol and number of evaluation episodes
- deterministic evaluation only:
	- epsilon strategy eval uses epsilon = 0
	- boltzmann strategy eval uses greedy argmax action (or extremely low T equivalent)
- same logging schema for both strategies

---
# **0. Principles For This Phase**

What we optimize for:
- simple code
- readable notebook
- low abstraction
- quick iteration
- solid outputs (tables, plots, conclusions)

What we avoid for now:
- new folder architecture work
- refactoring to src modules
- generic helper layers
- heavy engineering patterns

Implementation policy:
- code lives in notebook cells
- small helper functions are allowed in notebook
- if a function is used only once and short, keep inline
- if a function is reused across many cells, define once in an early section cell

---
# **1. Reuse Strategy From Task 2**

We follow task2.ipynb as the primary template.

## 1.1 Keep same setup skeleton
- imports, device detection, seed setup
- Colab/local path setup pattern
- FAST_RUN and LOAD_SAVED_RESULTS flags
- run budget constants

## 1.2 Keep same training/evaluation style
- keep train and evaluate function style close to Task 2
- keep rolling-average plots style
- keep save_result and persistent results JSON pattern
- keep clear print logs for run summary

## 1.3 Continue from best Task 2 config

Start Task 3 from Task 2 best baseline (as already defined in task2 notebook):
- BEST_ARCH_CLASS
- BEST_PREPROCESS
- BEST_N_FRAMES
- BEST_GAMMA
- BEST_LOSS_FN
- BEST_WARMUP
- BEST_HEURISTIC

Task 3 rule:
- backbone stays fixed
- exploration policy changes

Backbone lock checklist (must be printed in notebook before experiments):
- BEST_ARCH_CLASS
- BEST_PREPROCESS
- BEST_N_FRAMES
- BEST_GAMMA
- BEST_LOSS_FN
- BEST_WARMUP
- BEST_HEURISTIC
- replay buffer settings
- target update settings

---
# **2. Notebook Sections To Implement**

Use this exact flow in the Task 3 notebook.

---
# **0. Project Set Up**
Concise section for setup and controls.

## 0.1 Imports and Runtime Setup
- copy Task 2 setup style
- include device print
- include local/colab path logic

## 0.2 Constants and Run Controls
- FAST_RUN
- LOAD_SAVED_RESULTS
- N_EPISODES_EXP / N_EPISODES / N_EPISODES_LG style constants
- seed list for multi-seed runs

## 0.3 Fixed Backbone From Task 2
- set best config constants from Task 2
- print one compact backbone summary table

## 0.4 Environment Sanity Check
- instantiate difficulty 0 env
- verify obs/action shapes and one short smoke run

---
# **1. Results Tracker**
Simple and robust, similar to Task 2.

## 1.1 Load or Initialize Results
- load JSON if LOAD_SAVED_RESULTS=True and file exists
- otherwise create empty dict

## 1.2 save_result helper
- append result by key
- write JSON to disk immediately

## 1.3 Results display helper
- dataframe table sorted by main metric

## 1.4 Schema checks
- lightweight asserts only
- keep simple key checks for run integrity

---
# **2. Task 3 Method Setup**

## 2.1 Epsilon schedules
- define linear, exponential, and step options
- one plotting cell to visualize schedule shapes

## 2.2 Boltzmann schedules
- define temperature schedules
- one plotting cell to visualize schedule shapes

## 2.3 Action selection helpers
- epsilon-greedy action function
- boltzmann action function
- tiny sanity tests in notebook

Required Boltzmann implementation detail:
- use numerically stable softmax:
	- subtract max(Q) before exp
	- divide by max(temperature, small_eps)
	- normalize to sum 1

Required sanity checks:
- probabilities sum to 1
- no NaN or inf probabilities
- high temperature gives near-uniform probabilities
- low temperature gives near-greedy probabilities

---
# **3. Epsilon-Greedy Experiments**

## 3.1 Experiment grid
- choose 3 epsilon schedule configs
- keep seeds fixed across configs

## 3.2 Train runs
- run training loop with fixed backbone and chosen epsilon schedule
- track reward, loss, q, epsilon, entropy, wall clock

Primary budget rule:
- compare by equal training steps (or equal number of environment transitions), not equal wall-clock duration

## 3.3 Evaluate runs
- evaluate each trained model with same protocol

Evaluation rule:
- force deterministic action selection in evaluation loop

## 3.4 Save artifacts and rows
- save checkpoint and metrics summary
- write run record to results JSON

## 3.5 Analyze epsilon results
- curves and summary table
- select best epsilon variant

---
# **4. Boltzmann Experiments**

## 4.1 Experiment grid
- choose 3 temperature schedule configs
- use same seeds as epsilon experiments

## 4.2 Train runs
- same backbone, only strategy changes

## 4.3 Evaluate runs
- same eval protocol used for epsilon

Evaluation rule:
- force deterministic action selection in evaluation loop

## 4.4 Save artifacts and rows
- same result format for fair comparison

## 4.5 Analyze boltzmann results
- curves and summary table
- select best boltzmann variant

---
# **5. Cross-Strategy Comparison (Main Grading Section)**

This is the core answer to Task 3 grading.

## 5.1 Best-vs-best table
- best epsilon vs best boltzmann
- include mean and std across seeds

## 5.2 Learning curves overlay
- score
- loss
- mean q

## 5.3 Efficiency comparison
- score per hour
- q per hour

## 5.4 Exploration behavior comparison
- epsilon or temperature trajectories
- action entropy trajectory

## 5.5 Sensitivity summary
- which strategy is easier/harder to tune

Additional required comparison metrics:
- exploration diversity over training (action entropy)
- convergence speed by steps-to-target-relative-performance (not fixed absolute score threshold)
- final performance variance across seeds

---
# **6. Qualitative Results and Conclusions**

## 6.1 Best model rollout preview
- short frame strip or animation

## 6.2 Failure cases
- a small table: failure pattern and likely reason

## 6.3 Final recommendation
- winner strategy for this environment
- concise tradeoffs and caveats

---
# **7. Horizontal Build Plan (Simple Execution Loop)**

We implement in this order, one block at a time.

Step A:
- implement Sections 0 and 1
- run with FAST_RUN
- confirm setup and results tracker work

Step B:
- implement Section 2 helpers
- run sanity checks for schedules and action samplers

Step C:
- implement Section 3 epsilon runs
- run small experiment set
- validate outputs and results records

Step C.1:
- verify deterministic evaluation behavior by explicitly printing exploration parameters in eval mode

Step D:
- implement Section 4 boltzmann runs
- run small experiment set
- validate outputs and results records

Step D.1:
- verify softmax probabilities and sampling behavior with a standalone sanity cell

Step E:
- implement Section 5 comparison outputs
- verify table correctness and plot readability

Step F:
- implement Section 6 narrative and final visuals
- final end-to-end notebook run (FAST_RUN first)

---
# **8. Output Checklist (What Must Exist)**

## 8.1 Must-have tables
- fixed backbone summary
- epsilon experiment summary
- boltzmann experiment summary
- best-vs-best comparison
- final run inventory

## 8.2 Must-have plots
- epsilon schedule plot
- temperature schedule plot
- reward curves
- loss curves
- q-value curves
- entropy curves
- efficiency plot
- best-vs-best overlay

---
# **9. Tests Inside Notebook (No Src Tests For Now)**

Keep test style simple and local.

## 9.1 Sanity asserts
- obs and action shape checks
- schedule value bounds
- boltzmann probs sum to 1
- valid action ids
- no NaN/inf in probability vectors
- deterministic eval uses zero exploration

## 9.2 Tiny-run smoke tests
- very small episode count for each strategy
- confirm no crashes and valid metrics

## 9.3 Results integrity checks
- result record has required fields
- metric values are finite
- files save correctly
- train/eval step budgets match protocol

---
# **10. Notebook Task List (Simple)**

| ID    | Importance | Task                                         | Why                            |
| ----- | ---------- | -------------------------------------------- | ------------------------------ |
| T3-01 | 10         | Setup cell block copied from Task 2 style    | stable local/colab execution   |
| T3-02 | 10         | Fixed backbone block from Task 2 best config | continuity with previous phase |
| T3-03 | 10         | Results tracker block                        | no lost experiments            |
| T3-04 | 9          | Epsilon schedule helpers + plots             | strategy A setup               |
| T3-05 | 9          | Boltzmann schedule helpers + plots           | strategy B setup               |
| T3-06 | 10         | Epsilon experiment loop                      | graded strategy implementation |
| T3-07 | 10         | Boltzmann experiment loop                    | graded strategy implementation |
| T3-08 | 10         | Cross-strategy comparison tables and plots   | graded analysis section        |
| T3-09 | 8          | Qualitative rollout + failure analysis       | stronger presentation          |
| T3-10 | 10         | Final concise conclusion section             | direct grading alignment       |

---
# **Doubts**

Keeping existing doubts, plus one new one specific to notebook-only implementation.

## !!!!! D1. Final comparison budget: fixed steps or fixed wall-clock time?
Option A:
- fixed training steps

Option B:
- fixed wall-clock budget

Current lean:
- Option A main analysis + report wall-clock efficiency

## !!!!! D2. How many schedule variants per strategy in the main notebook?
Option A:
- 3 variants per strategy

Option B:
- 4 to 6 variants per strategy

Current lean:
- Option A in main body

## !!!!! D3. Frame stacking in Task 3?
Option A:
- keep Task 2 winner (likely 4 frames)

Option B:
- force single frame for simpler interpretation

Current lean:
- keep Task 2 winner fixed

## !!!!! D4. Harder difficulty in main conclusion?
Option A:
- difficulty 0 main, harder as bonus

Option B:
- include harder difficulty in main section

Current lean:
- Option A

## !!!!! D5. Number of seeds in main comparison?
Option A:
- 3

Option B:
- 5

Current lean:
- 5 if runtime allows, else 3

Plan decision:
- use 3 seeds for tuning grid
- use 5 seeds for final best-vs-best if within budget
- if only 3 seeds are feasible for final, report that as a limitation explicitly

## !!!!! D6. Keep one notebook for Task 3 work or continue directly in task2.ipynb?
Option A:
- create/use dedicated task3 notebook
- cleaner final deliverable

Option B:
- continue in task2.ipynb and keep adding sections
- faster for development now

Current lean:
- Option A for cleaner grading and lower risk of presentation clutter

## !!!!! D7. Final comparative metric for convergence speed
Option A:
- steps to reach a fixed absolute score threshold

Option B:
- steps to reach a relative target (example: 80% of each strategy's best mean final score)

Current lean:
- Option B (more robust across run variance and schedule choices)
