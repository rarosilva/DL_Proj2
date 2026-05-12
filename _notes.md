# task1_organized.ipynb -- Precise Fix List for AI Agent
# Each task is self-contained, numbered, with exact cell location and exact change.
# The agent should NOT change anything not listed here.

---

## ANALYSIS SUMMARY (read before doing any task)

### What is correct and should NOT be touched
- Section 0: imports, device, FAST_RUN flag, constants -- all good
- Section 1: env tests, channel decomp, semantic obs, reward/episode tests, baselines -- all good
- Section 2: preprocess_obs + preprocess_obs_r_channel, train(), evaluate(), heuristic functions -- all good
- Section 3.2: multi-architecture comparison (DQN_Adam, DQN_RMS, BiggerDQN_Adam, BiggerDQN_RMS, StridedDQN_RMS) -- good
- Section 3.3: R-channel test -- good (correctly uses preprocess_fn param)
- Section 3.5: dropout ablation -- good
- Section 3.6: loss/reward_clip ablation -- good
- Section 3.7: gamma ablation -- good
- Section 4: heuristics + warmup -- good
- Section 5: BC collect, train, compare -- good structure
- Section 6.3 + 6.4: Agent class + local_eval -- good

### What is WRONG or MISSING (all tasks below fix these)

---

## TASK 1 -- Fix: DQN architecture uses MaxPool + AdaptiveMaxPool -- should be ablation only, not baseline

CELL: Cell 35 (DQN class definition)

PROBLEM:
The DQN used everywhere as the "standard" model uses:
- MaxPool2d(2,2) twice -> aggressively discards spatial position
- AdaptiveMaxPool2d((2,2)) in classifier -> squashes to 2x2 = 16 values per channel

For a positional game (ship must dodge debris at exact grid positions), this is a known weakness.
The current code treats MaxPool DQN as the baseline used in ALL ablation experiments (sections
3.3, 3.4, 3.5, 3.6, 3.7, 4, 5, 6). This is fine AS a baseline to show the limitation.
But the markdowns in 3.1 must clearly say this is the baseline and why it's expected to be weaker
than StridedDQN. Currently the 3.1 markdown briefly mentions MaxPool is a "known issue" but does
not clearly state that DQN is used as the canonical baseline precisely to then show StridedDQN improves it.

FIX: Edit the markdown in Cell 34 (Section 3.1 header) to add one short paragraph:
"DQN is our canonical baseline -- we use it in ALL subsequent ablations (Sections 3.3-3.7, 4, 5, 6)
to keep comparisons fair. StridedDQN is tested in Section 3.2 to show the MaxPool limitation.
If StridedDQN wins, the 'best config' cell in Section 6.1 should use it instead."

No code change needed. Markdown edit only.

---

## TASK 2 -- Fix: ReLU vs LeakyReLU -- DQN uses ReLU, BiggerDQN uses LeakyReLU -- inconsistency not explained

CELL: Cell 34 (Section 3.1 markdown) and Cell 35 (DQN class)

PROBLEM:
DQN uses nn.ReLU(). BiggerDQN uses nn.LeakyReLU(0.01). StridedDQN uses nn.ReLU().
The markdown in 3.1 never explains this difference or whether we should test ReLU vs LeakyReLU on DQN.

For online DQN (no replay buffer), dead ReLU neurons are more likely because:
- updates are noisy
- some neurons may never fire again after a bad update
LeakyReLU (0.01 negative slope) avoids this completely.

FIX: In Cell 34 markdown (Section 3.1), add a short note:
"DQN uses ReLU. BiggerDQN and GroupsDQN use LeakyReLU(0.01) because deeper nets suffer more from
dead neurons (a neuron whose input is always negative -> always outputs 0 -> zero gradient).
For the small baseline DQN (2 layers), ReLU is fine. For the deeper variants, LeakyReLU is safer."

No code change needed. Markdown edit only.

---

## TASK 3 -- Fix: Section 3.4 (frame stacking) uses N_EPISODES (small), not N_EPISODES_MD

CELL: Cell 44 (frame stacking experiment)

PROBLEM:
Frame stacking cells 44 uses N_EPISODES for the DQN runs and N_EPISODES for GroupsDQN.
But Section 3.2 (architecture comparison) uses N_EPISODES_MD.
This makes frame stacking results not comparable to architecture results (fewer episodes = noisier).

FIX: In Cell 44, change BOTH train() calls to use n_episodes=N_EPISODES_MD (not N_EPISODES).

Exact changes:
Line: `train(envs['difficulty_0'], dqn, opt, n_episodes=N_EPISODES, n_frames=sf,`
Change to: `train(envs['difficulty_0'], dqn, opt, n_episodes=N_EPISODES_MD, n_frames=sf,`

Line: `train(envs['difficulty_0'], gdqn, gopt, n_episodes=N_EPISODES, n_frames=sf,`
Change to: `train(envs['difficulty_0'], gdqn, gopt, n_episodes=N_EPISODES_MD, n_frames=sf,`

Also update the save_result calls accordingly:
`save_result(f'frames/DQN_{sf}', ms, ss, q_val=mq, n_episodes=N_EPISODES)`
-> `save_result(f'frames/DQN_{sf}', ms, ss, q_val=mq, n_episodes=N_EPISODES_MD)`

`save_result(f'frames/GroupsDQN_{sf}', ms, ss, q_val=mq, n_episodes=N_EPISODES)`
-> `save_result(f'frames/GroupsDQN_{sf}', ms, ss, q_val=mq, n_episodes=N_EPISODES_MD)`

---

## TASK 4 -- Fix: No "best config selection" cell -- ablation results are never consolidated into a best config used for Section 5 and 6

CELL: Add new code cell AFTER Cell 54 (Section 3.8) and BEFORE Cell 55 (Section 4 header)

PROBLEM:
Sections 3.2-3.7 run many experiments. Section 5 (BC) and Section 6 (final training) just
hardcode DQN with n_frames=1. There is no cell that looks at all_results and picks the best
architecture, best n_frames, best preprocess (RGB vs R), etc.
The reader cannot tell how the final config was chosen.

FIX: Add a new code cell after Cell 54 with this content:

```python
# -- BEST CONFIG SELECTION --
# Read experiment results to pick the best setting for Section 5 (BC) and Section 6 (final).
# This is MANUAL: look at the tables above and set these variables.
# We do NOT auto-select because ablations are short (FAST_RUN) -- use your judgment.

# Set these after reviewing Section 3 tables:
BEST_ARCH_CLASS   = DQN          # e.g. DQN or StridedDQN (from Section 3.2 table)
BEST_N_FRAMES     = 1            # e.g. 1, 2, or 4 (from Section 3.4 table)
BEST_PREPROCESS   = preprocess_obs  # preprocess_obs or preprocess_obs_r_channel (Section 3.3)
BEST_N_CHANNELS   = 3            # 3 if using preprocess_obs, 1 if using r_channel
BEST_WARMUP       = 0.7          # from Section 4.6 warmup table
BEST_HEURISTIC    = heuristic_policy  # from Section 4.5 comparison

print("Best config for Sections 5 and 6:")
print(f"  arch:        {BEST_ARCH_CLASS.__name__}")
print(f"  n_frames:    {BEST_N_FRAMES}")
print(f"  preprocess:  {BEST_PREPROCESS.__name__}")
print(f"  n_channels:  {BEST_N_CHANNELS}")
print(f"  warmup_pct:  {BEST_WARMUP}")
print(f"  heuristic:   {BEST_HEURISTIC.__name__}")
```

Also add a short markdown cell before this code cell:
```markdown
---
## 3.9 Best Config Selection
After reviewing all Section 3 experiment tables, we manually set the best configuration.
This config is used in Section 5 (Behavioral Cloning) and Section 6 (Final Training).
Set the variables in the cell below before running Sections 5-6.
```

---

## TASK 5 -- Fix: Section 5 (BC) and Section 6 (final) hardcode DQN + n_frames=1 instead of using BEST_* variables

CELLS: Cell 68 (bc_train), Cell 70 (BC comparison), Cell 73 (final training), Cell 74 (diff1 training)

PROBLEM:
After Task 4 adds BEST_ARCH_CLASS and BEST_N_FRAMES, these cells still hardcode DQN + n_frames=1.

FIX in Cell 68 (bc_train function): change the bc_net creation line from:
`bc_net = DQN(n_frames=1, dropout=0.0).to(device)`
to:
`bc_net = BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)`

FIX in Cell 70 (BC comparison loop): change this line:
`net = copy.deepcopy(bc_net) if use_bc else DQN(n_frames=1, dropout=0.0).to(device)`
to:
`net = copy.deepcopy(bc_net) if use_bc else BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)`

And change both train() calls in Cell 70 from:
`train(envs['difficulty_0'], net, opt, n_episodes=N_EPISODES, n_frames=1, ...`
to:
`train(envs['difficulty_0'], net, opt, n_episodes=N_EPISODES, n_frames=BEST_N_FRAMES, ...`

And both evaluate() calls from:
`evaluate(envs['difficulty_0'], net, 1, ...`
to:
`evaluate(envs['difficulty_0'], net, BEST_N_FRAMES, ...`

FIX in Cell 73 (final training): change:
`final_net = DQN(n_frames=1, dropout=0.0).to(device)`
to:
`final_net = BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)`

And change both train() calls (in Cell 73 and Cell 74) from `n_frames=1` to `n_frames=BEST_N_FRAMES`.
And the evaluate() calls from `final_net, 1,` to `final_net, BEST_N_FRAMES,`.

Also add `preprocess_fn=BEST_PREPROCESS` to all train() and evaluate() calls in Cells 73 and 74.

FIX in Cell 73: also update the markdown Section 6.1 (Cell 72) to say:
"Uses the best configuration selected in Section 3.9 (arch, n_frames, preprocess, warmup)."

---

## TASK 6 -- Fix: BC section (Cell 70) trains each condition for N_EPISODES (too short) -- results are noisy and not comparable to Section 3

CELL: Cell 70 (BC comparison)

PROBLEM:
BC comparison uses N_EPISODES. Section 3 ablations use N_EPISODES_MD (2x longer).
With very few episodes in FAST_RUN (5 vs 10), BC results are too noisy to interpret.

FIX: Change the train() call in Cell 70 from `n_episodes=N_EPISODES` to `n_episodes=N_EPISODES_MD`.
And the evaluate() timing is also wrong: `start = time.time()` is BEFORE `evaluate()` but training
already ran before it -- the training time is not captured. Fix the timing pattern to:

```python
for label, use_bc, w in bc_conditions:
    net = copy.deepcopy(bc_net) if use_bc else BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)
    opt = optim.RMSprop(net.parameters(), lr=LR, alpha=0.99, eps=1e-8)
    start = time.time()                            # <- start timing HERE before train()
    _, rewards, losses, qs = train(
        envs['difficulty_0'], net, opt, n_episodes=N_EPISODES_MD, n_frames=BEST_N_FRAMES,
        heuristic=BEST_HEURISTIC, warmup_pct=w, use_huber=True, name=label,
        preprocess_fn=BEST_PREPROCESS)
    t = time.time() - start                        # <- capture training time
    mean, std, mq, s_per_t, _ = evaluate(envs['difficulty_0'], net, BEST_N_FRAMES, t,
                                          preprocess_fn=BEST_PREPROCESS)
    ...
```

---

## TASK 7 -- Fix: assert vs print -- replace bare prints with asserts for correctness checks

The following specific print("[ok] ...") lines should become assert + print pairs
so the notebook actually halts on failure instead of silently passing.

### Cell 6 (env creation):
Change:
```python
print("[ok] env shapes confirmed")
```
To:
```python
assert envs["difficulty_0"].observation_space.shape == (54, 39, 3), \
    f"unexpected obs shape: {envs['difficulty_0'].observation_space.shape}"
assert envs["difficulty_0"].action_space.n == 2, \
    f"unexpected action count: {envs['difficulty_0'].action_space.n}"
print("[ok] env shapes confirmed")
```

### Cell 16 (reward structure):
The current cell already does comparisons but uses print, not assert.
Change the reward comparison prints to assert:
```python
assert abs(r_up   -  0.02) < 0.01, f"unexpected up reward: {r_up}"
assert abs(r_down - (-0.01)) < 0.01, f"unexpected down reward: {r_down}"
print(f"move up reward:   {r_up:.4f}   (expected ~+0.02)  [ok]")
print(f"move down reward: {r_down:.4f}  (expected ~-0.01)  [ok]")
```

### Cell 18 (episode length):
Current code does NOT assert step_count == 600.
Add after the while loop:
```python
assert step_count == 600, f"expected 600 steps, got {step_count}"
assert not terminated, "terminated should never be True"
print(f"[ok] episode ended at step {step_count} (expected 600)")
```

### Cell 24 (preprocess):
Change the shape prints to assert+print:
```python
assert out_rgb.shape == (1, 3, 54, 39), f"RGB preprocess shape wrong: {out_rgb.shape}"
assert out_r.shape   == (1, 1, 54, 39), f"R-channel preprocess shape wrong: {out_r.shape}"
assert out_rgb.min() >= 0.0 and out_rgb.max() <= 1.0, "RGB values not normalized to [0,1]"
print(f"[ok] preprocess_obs:           {obs_test.shape} -> {out_rgb.shape}")
print(f"[ok] preprocess_obs_r_channel: {obs_test.shape} -> {out_r.shape}")
```

### Cell 36 (architecture sanity check):
Currently prints but does not assert.
Add inside the loop:
```python
assert out.shape == (1, 2), f"{cls.__name__}: expected output (1,2), got {out.shape}"
```
(there is already an assert-like check in the loop -- confirm it's actually an assert not a print)

### Cell 37 (smoke test):
Change `print("[ok] smoke test passed")` to also assert mean_s >= 0:
```python
assert mean_s >= 0, f"smoke test: score should be non-negative, got {mean_s}"
print(f"[ok] smoke test passed: score={mean_s:.2f}  q={mean_q:.4f}")
```

### Cell 68 (BC accuracy):
Already has assert -- keep as-is. No change needed.

---

## TASK 8 -- Fix: input normalization -- confirm it is done and add a short note

CELL: Cell 23 markdown (Section 2.2) and Cell 24 (preprocess_obs)

PROBLEM:
Input normalization (dividing by 255.0 to get [0,1]) IS done in preprocess_obs.
But the markdown in Cell 23 never explicitly states WHY it is important.

FIX: In Cell 23 markdown, add one short bullet:
"- **Input normalization**: divide uint8 values (0-255) by 255.0 -> float32 in [0,1].
  Why: neural net weights are initialized for ~unit-scale inputs. Raw [0,255] values
  would make the first layer's activations ~100x too large -> large gradients -> divergence.
  This is NOT optional."

No code change needed. Markdown edit only.

---

## TASK 9 -- Fix: gradient clipping -- confirm it is in train() and add explanation in markdown

CELL: Cell 25 (Section 2.3 markdown) and Cell 26 (train() code)

The code in Cell 26 DOES include:
`torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)`
This is correct. But the markdown in Cell 25 (the train() design note) does not mention gradient clipping.

FIX: In Cell 25 markdown, in the "How train() works" pseudocode block, add:
`clip_grad_norm_(params, GRAD_CLIP_NORM)  # cap gradient L2-norm to avoid exploding gradients`
as a comment inside the backprop step description.

No code change needed. Markdown edit only.

---

## TASK 10 -- Fix: Section 3.4 frame stacking results table (Cell 45) has wrong column logic

CELL: Cell 45 (frame stacking results table)

PROBLEM:
The table cell uses:
`arch_name = 'DQN' if k.startswith('DQN_') else 'GroupsDQN'`
The keys in sf_results are like 'DQN_1', 'DQN_2', 'DQN_4', 'GroupsDQN_2', 'GroupsDQN_4'.
This logic is correct, but 's_per_t' is not stored in sf_results (only score, std, q, n_frames).
The table tries to access v['s_per_t'] which will KeyError.

FIX: In Cell 44, update sf_results storage to also capture s_per_t:
Change:
`sf_results[f'DQN_{sf}'] = {'n_frames': sf, 'score': ms, 'std': ss, 'q': mq}`
To:
`sf_results[f'DQN_{sf}'] = {'n_frames': sf, 'score': ms, 'std': ss, 'q': mq, 's_per_t': spt}`

And capture spt from evaluate:
Change:
`ms, ss, mq, _, _ = evaluate(envs['difficulty_0'], dqn, sf, time.time()-start)`
To:
`ms, ss, mq, spt, _ = evaluate(envs['difficulty_0'], dqn, sf, time.time()-start)`

Same fix for the GroupsDQN branch:
`ms, ss, mq, _, _ =` -> `ms, ss, mq, spt, _ =`
and store `'s_per_t': spt` in sf_results.

Then in Cell 45, update the row building to use v.get('s_per_t', float('nan')):
```python
rows.append({
    "Config":  k,
    "Arch":    arch_name,
    "n_frames": v["n_frames"],
    "Score":   f"{v['score']:.2f}+-{v['std']:.2f}",
    "Q-val":   f"{v['q']:.4f}",
    "Score/s": f"{v.get('s_per_t', float('nan')):.4f}",
})
```

---

## TASK 11 -- Fix: Section 6 final results summary (Cell 76) does not include BC results

CELL: Cell 76 (final results summary)

PROBLEM:
The summary table and bar chart in Cell 76 include random, always_up, heuristics, and DQN_final.
It does NOT include BC comparison results even though Section 5 computed them.
The reader cannot see how BC compares to the final DQN.

FIX: Add BC results to summary_rows after the heuristic rows:
```python
if bc_comparison:  # only if Section 5 was run
    best_bc_label = max(bc_comparison, key=lambda k: bc_comparison[k]['mean'])
    row = {'Policy': f'BC_best ({best_bc_label})'}
    for i in range(4): row[f'Diff{i}'] = '-'
    row['Diff0'] = f"{bc_comparison[best_bc_label]['mean']:.2f}+-{bc_comparison[best_bc_label]['std']:.2f}"
    summary_rows.append(row)
```

Also add 'BC_best' to the _pol_names list in the bar chart section with corresponding data extraction.

---

## TASK 12 -- Fix: BC section markdown (Cell 64) does not explain the difference between BC and warmup_pct

CELL: Cell 64 (Section 5 header markdown)

PROBLEM:
The markdown explains BC well but does not explicitly contrast it with warmup_pct.
A reader who just read Section 4 will wonder: "isn't this basically the same as warmup=1.0?"

FIX: Add the following to Cell 64 markdown:
```
### BC vs warmup_pct: what is the difference?

warmup_pct in train(): at each step, with probability w, use heuristic action.
The heuristic ACTION is used but the observation comes from wherever the RL agent has navigated.
The RL network sees RGB observations from trajectories it partially controls.

BC pretraining: the DQN is first trained via supervised learning on (RGB obs, heuristic action) pairs.
The observations come entirely from heuristic-controlled trajectories (different distribution).
After BC, RL fine-tuning starts from a network that already knows the rough mapping RGB->action.

Key difference: BC gives a smarter weight initialization. warmup gives better transitions during RL.
They can be combined (BC init + warmup_pct > 0 during RL), which is what 'BC+warmup=0.5' tests.
```

---

## TASK 13 -- Fix: human play for BC data collection -- answer the question and add a note

CELL: Cell 65 markdown (Section 5.1 header)

QUESTION ASKED: can a human play the game and use those frames for BC?
ANSWER: yes, but space_race_env.py requires a pygame display (manual_play.py shows this).
In a Colab/notebook context there is no display -- it would crash.
The heuristic is the practical alternative that works headlessly.

FIX: Add to Cell 65 markdown:
```
### Could we use human gameplay instead of the heuristic?
Yes in principle: play via `SpaceRace/manual_play.py` (requires pygame + display),
record (obs, action) pairs, save to file, load here.
In practice: Colab has no display -> pygame crashes. The heuristic is a headless alternative
that is repeatable and deterministic (same seed = same trajectory).
If you want human data: play locally, save with pickle/numpy, upload to Colab.
For this notebook: heuristic demonstrations are sufficient and more practical.
```

---

## TASK 14 -- Fix: Section 3.8 instability analysis (Cell 54) does not actually plot instability

CELL: Cell 54 (Section 3.8 summary)

PROBLEM:
Cell 54 prints architecture comparison tables but has a comment saying:
"# instability signal: rolling std of reward..." but does NOT actually plot it.
This is explicitly mentioned as important in the assignment (show Phase 1 instability
to motivate Phase 2 target network + replay buffer).

FIX: Add a rolling-std-of-reward plot AFTER the table in Cell 54.
The reward_history data is available from the last run in Cell 39 (arch comparison loop),
but it is not stored per-model. The simplest fix: re-run ONE model and capture reward_history,
then plot rolling std.

Add this code after the table print in Cell 54:
```python
# instability demonstration: re-run best arch, capture reward history, plot rolling std
_inst_net = DQN(n_frames=1, dropout=0.0).to(device)
_inst_opt = optim.RMSprop(_inst_net.parameters(), lr=LR, alpha=0.99, eps=1e-8)
_, _inst_rewards, _inst_losses, _ = train(
    envs['difficulty_0'], _inst_net, _inst_opt,
    n_episodes=N_EPISODES_MD, n_frames=1,
    heuristic=heuristic_policy, warmup_pct=0.7, use_huber=True,
    name='instability demo')

window = max(3, N_EPISODES_MD // 10)
rolling_std = [np.std(_inst_rewards[max(0, i-window):i+1]) for i in range(len(_inst_rewards))]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(_inst_rewards, alpha=0.6, label='episode reward')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
axes[0].set_title('Online DQN reward trajectory (Phase 1 -- no buffer, no target net)')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(rolling_std, color='orange', label=f'rolling std (window={window})')
axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Reward std (rolling)')
axes[1].set_title('Training instability -- high variance = unstable')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
print("High reward variance = Phase 1 instability. Phase 2 (replay buffer + target net) fixes this.")
```

---

## TASK 15 -- Fix: Section 6 final training saves to '../model0.pt' -- wrong path

CELL: Cell 73 (final training)

PROBLEM:
`torch.save(final_net.state_dict(), '../model0.pt')`
The relative path '../model0.pt' depends on where the notebook is run from.
In Colab it may not exist. In VS Code it may save to the wrong directory.
The Agent class in Cell 78 loads from `os.path.join(os.getcwd(), 'model.pt')` -- the cwd.

FIX: Change save paths in Cell 73 and Cell 74 to use os.getcwd():
```python
# Cell 73:
_save_path = os.path.join(os.getcwd(), 'model0.pt')
torch.save(final_net.state_dict(), _save_path)
print(f'[ok] model0.pt saved -> {_save_path}')

# Cell 74:
_save_path1 = os.path.join(os.getcwd(), 'model1.pt')
_save_path_sub = os.path.join(os.getcwd(), 'model.pt')
torch.save(final_net.state_dict(), _save_path1)
torch.save(final_net.state_dict(), _save_path_sub)
print(f'[ok] model1.pt + model.pt saved -> {_save_path_sub}')
```

Also update Section 6.1 markdown (Cell 72) to note:
"model.pt is saved to os.getcwd() -- the directory where this notebook is running.
The Agent class in Section 6.3 loads from the same path."

---

## TASK 16 -- Fix: Section 6.2 summary bar chart (Cell 76) does not include error bars

CELL: Cell 76

PROBLEM:
The bar chart shows mean scores for each policy but has no error bars (std).
Without error bars, it's impossible to tell if DQN actually beats the heuristic or if
the difference is within noise. For the presentation, this is important.

FIX: Add yerr to the bar() calls:
```python
# collect stds alongside means
_stds_d0, _stds_d1 = [], []
for _pol in _pol_names:
    if _pol == 'DQN_final':
        _stds_d0.append(std0); _stds_d1.append(std1)
    elif _pol.startswith('heuristic_'):
        _hn = _pol.split('_', 1)[1]
        _stds_d0.append(heuristic_results[_hn][0][1])
        _stds_d1.append(heuristic_results[_hn][1][1])
    else:
        _stds_d0.append(baseline_results[_pol][0][1])
        _stds_d1.append(baseline_results[_pol][1][1])

bars0 = ax.bar(_x - _w/2, _scores_d0, _w, yerr=_stds_d0, capsize=4,
               label='Difficulty 0', color=_colors_d0, alpha=0.85)
bars1 = ax.bar(_x + _w/2, _scores_d1, _w, yerr=_stds_d1, capsize=4,
               label='Difficulty 1', color=_colors_d1, alpha=0.85)
```

---

## TASK 17 -- Add: Section 6.2 should also have a learning curves dashboard

CELL: Add new code cell AFTER Cell 76 (results table) and BEFORE Cell 77 (Agent section)

PROBLEM:
The assignment rubric explicitly asks for "visualizations of learning curves".
The individual train() calls each show a reward+Q plot, but there's no consolidated
learning curve view in the final section. The grader will want to see Phase 1 instability
clearly demonstrated in the final section.

FIX: Add a new code cell after Cell 76:
```python
# ── Final learning curves dashboard ──────────────────────────────
# rew0, q0, los0 from Cell 73 (diff0 training)
# rew1, q1, los1 from Cell 74 (diff1 training)

if 'rew0' in dir() and 'rew1' in dir():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # reward
    axes[0,0].plot(rew0, alpha=0.4, label='diff0 raw')
    _avg0 = [np.mean(rew0[max(0,i-10):i+1]) for i in range(len(rew0))]
    axes[0,0].plot(_avg0, label='diff0 rolling avg (10ep)')
    axes[0,0].set_title('Reward -- Difficulty 0'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(rew1, alpha=0.4, label='diff1 raw')
    _avg1 = [np.mean(rew1[max(0,i-10):i+1]) for i in range(len(rew1))]
    axes[0,1].plot(_avg1, label='diff1 rolling avg (10ep)')
    axes[0,1].set_title('Reward -- Difficulty 1'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # loss
    axes[1,0].plot(los0, alpha=0.6, color='orange', label='diff0 loss')
    axes[1,0].set_title('Loss -- Difficulty 0'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(los1, alpha=0.6, color='orange', label='diff1 loss')
    axes[1,1].set_title('Loss -- Difficulty 1'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    for ax in axes.flat: ax.set_xlabel('Episode')
    plt.suptitle('Final Training Learning Curves (Phase 1 -- Basic DQN)', fontsize=12)
    plt.tight_layout(); plt.show()
else:
    print("[!] Run Section 6.1 first to generate rew0, rew1, los0, los1")
```

---

## TASK 18 -- Fix: Section 3.5 BN ablation (Cell 48) -- DQN_BN class has no docstring and markdown doesn't explain expected result clearly

CELL: Cell 46 (Section 3.5 markdown) and Cell 48 (DQN_BN code)

PROBLEM:
The markdown in Cell 46 explains why BN hurts but does not say what specific failure mode to look for
(training divergence / NaN loss / near-zero scores).

FIX: In Cell 46 markdown, add one line:
"Expected outcome: DQN_BN achieves near-zero score OR shows NaN/exploding loss.
If it trains stably, that means the env happens to provide enough variance in a single step
to keep BN stable -- an edge case. Either result is worth noting."

In Cell 48, add a brief docstring to DQN_BN:
```python
class DQN_BN(nn.Module):
    """DQN with BatchNorm2d. Expected to hurt: BN with batch_size=1 has undefined std.
    Included as ablation to confirm our decision NOT to use BN in baseline DQN."""
```

---

## PRIORITY ORDER

Run these tasks in order. Tasks 1-3, 7-9 are markdown/small fixes (quick).
Tasks 4-6 are the most important (best config flow). Tasks 10-18 are improvements.

| Priority | Task | Type    | Impact                                        |
| -------- | ---- | ------- | --------------------------------------------- |
| 1        | T7   | code    | correctness -- asserts catch real errors      |
| 2        | T10  | code    | bug -- KeyError on s_per_t                    |
| 3        | T15  | code    | bug -- wrong save path for model.pt           |
| 4        | T4   | code+md | new cell -- best config selection             |
| 5        | T5   | code    | correctness -- BC/final use best config       |
| 6        | T6   | code    | correctness -- BC timing + episode count      |
| 7        | T3   | code    | fairness -- frame stacking uses N_EPISODES_MD |
| 8        | T16  | code    | presentation -- error bars on bar chart       |
| 9        | T17  | code    | presentation -- learning curves dashboard     |
| 10       | T14  | code    | presentation -- instability plot              |
| 11       | T11  | code    | completeness -- BC results in summary         |
| 12       | T1   | md      | clarity -- baseline DQN rationale             |
| 13       | T2   | md      | clarity -- ReLU vs LeakyReLU explanation      |
| 14       | T8   | md      | clarity -- normalization rationale            |
| 15       | T9   | md      | clarity -- gradient clipping in pseudocode    |
| 16       | T12  | md      | clarity -- BC vs warmup difference            |
| 17       | T13  | md      | clarity -- human play answer                  |
| 18       | T18  | md+code | clarity -- BN expected failure mode           |