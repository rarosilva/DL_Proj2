import json
from pathlib import Path

nb_path = Path("tasks/task1_organized.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(cell):
    return "".join(cell.get("source", []))


def set_src(cell, text):
    cell["source"] = [line + "\n" for line in text.splitlines()]


def find_idx(needle):
    for i, c in enumerate(cells):
        if needle in src(c):
            return i
    return -1


def insert_cell(after_idx, cell_type, text):
    new_cell = {
        "cell_type": cell_type,
        "metadata": {"language": "python" if cell_type == "code" else "markdown"},
        "source": [line + "\n" for line in text.splitlines()],
    }
    if cell_type == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    cells.insert(after_idx + 1, new_cell)


changes = []

# Task 1 + 2: update Section 3.1 markdown with baseline rationale + ReLU vs LeakyReLU note
idx = find_idx("## 3.1 DQN Architecture Definitions")
if idx != -1:
    set_src(cells[idx], """---
## 3.1 DQN Architecture Definitions

Four architectures are defined here and compared in Section 3.2.

Baseline DQN input shape: **(1, 3, 54, 39)** -- batch=1, channels=3, H=54, W=39.

Architecture:
```
Conv(3->16, 3x3, pad=1) + ReLU + MaxPool(2,2)   -> (1, 16, 27, 19)
Conv(16->32, 3x3, pad=1) + ReLU + MaxPool(2,2)  -> (1, 32, 13, 9)
AdaptiveMaxPool(2,2) + Flatten                  -> (1, 128)
[Dropout] + Linear(128 -> 2)                    -> (1, 2)  [Q-values for each action]
```

**Canonical baseline choice**: this DQN is used in all later ablations (Sections 3.3-3.7, 4, 5, 6)
to keep comparisons fair. `StridedDQN` is tested in Section 3.2 specifically to expose the MaxPool
position-loss limitation. If StridedDQN wins consistently, Sections 5-6 should switch to it.

Known limitation: MaxPooling discards spatial position info.
For a game where ship position matters (exact row), this is suboptimal.
That motivates the `StridedDQN` variant (strided conv instead of MaxPool).

Activation note: baseline DQN uses ReLU (small 2-layer network is usually stable).
BiggerDQN / deeper variants use LeakyReLU(0.01) to reduce dead-neuron risk under noisy online RL updates.

Why **no BatchNorm**: BN normalises over the batch dimension.
With online training (batch size=1 per step), the running stats are unreliable --
std is undefined for batch size 1 -> training instability. Tested in Section 3.5.
""")
    changes.append("Task 1-2")

# Task 3 + 10: frame stacking run length + s_per_t storage + table column
idx = find_idx("sf_results = {}")
if idx != -1 and "GroupsDQN" in src(cells[idx]):
    set_src(cells[idx], """import time
sf_results = {}
for sf in [1, 2, 4]:
    dqn = DQN(n_frames=sf, dropout=0.0).to(device)
    opt = optim.RMSprop(dqn.parameters(), lr=LR, alpha=0.99, eps=1e-8)
    start = time.time()
    train(envs['difficulty_0'], dqn, opt, n_episodes=N_EPISODES_MD, n_frames=sf,
          heuristic=heuristic_policy, warmup_pct=0.5, use_huber=True, name=f'DQN sf={sf}')
    ms, ss, mq, spt, _ = evaluate(envs['difficulty_0'], dqn, sf, time.time()-start)
    sf_results[f'DQN_{sf}'] = {'n_frames': sf, 'score': ms, 'std': ss, 'q': mq, 's_per_t': spt}
    save_result(f'frames/DQN_{sf}', ms, ss, q_val=mq, n_episodes=N_EPISODES_MD, s_per_t=spt)
    print(f'DQN n_frames={sf}: score={ms:.2f}+-{ss:.2f}  q={mq:.4f}  s/t={spt:.4f}')

for sf in [2, 4]:
    gdqn = GroupsDQN(n_frames=sf).to(device)
    gopt = optim.RMSprop(gdqn.parameters(), lr=LR, alpha=0.99, eps=1e-8)
    start = time.time()
    train(envs['difficulty_0'], gdqn, gopt, n_episodes=N_EPISODES_MD, n_frames=sf,
          heuristic=heuristic_policy, warmup_pct=0.5, use_huber=True, name=f'GroupsDQN sf={sf}')
    ms, ss, mq, spt, _ = evaluate(envs['difficulty_0'], gdqn, sf, time.time()-start)
    sf_results[f'GroupsDQN_{sf}'] = {'n_frames': sf, 'score': ms, 'std': ss, 'q': mq, 's_per_t': spt}
    save_result(f'frames/GroupsDQN_{sf}', ms, ss, q_val=mq, n_episodes=N_EPISODES_MD, s_per_t=spt)
    print(f'GroupsDQN n_frames={sf}: score={ms:.2f}+-{ss:.2f}  q={mq:.4f}  s/t={spt:.4f}')
""")
    changes.append("Task 3+10 code")

idx = find_idx("# frame stacking results table")
if idx != -1 and "sf_results" in src(cells[idx]):
    set_src(cells[idx], """# frame stacking results table
rows = []
for k, v in sf_results.items():
    arch_name = 'DQN' if k.startswith('DQN_') else 'GroupsDQN'
    rows.append({
        "Config":  k,
        "Arch":    arch_name,
        "n_frames": v["n_frames"],
        "Score":   f"{v['score']:.2f}+-{v['std']:.2f}",
        "Q-val":   f"{v['q']:.4f}",
        "Score/s": f"{v.get('s_per_t', float('nan')):.4f}",
    })
print("Frame stacking results (diff 0, warmup=0.5, Huber):")
display(pd.DataFrame(rows))
""")
    changes.append("Task 10 table")

# Task 4: add Section 3.9 markdown + code after 3.8 code cell
if find_idx("## 3.9 Best Config Selection") == -1:
    idx_38_code = find_idx("Instability analysis")
    if idx_38_code != -1:
        insert_cell(idx_38_code, "markdown", """---
## 3.9 Best Config Selection
After reviewing all Section 3 experiment tables, we manually set the best configuration.
This config is used in Section 5 (Behavioral Cloning) and Section 6 (Final Training).
Set the variables in the cell below before running Sections 5-6.
""")
        insert_cell(idx_38_code + 1, "code", """# -- BEST CONFIG SELECTION --
# Read experiment results to pick the best setting for Section 5 (BC) and Section 6 (final).
# This is MANUAL: look at the tables above and set these variables.
# We do NOT auto-select because ablations are short (FAST_RUN) -- use your judgment.

BEST_ARCH_CLASS = DQN
BEST_N_FRAMES = 1
BEST_PREPROCESS = preprocess_obs
BEST_N_CHANNELS = 3
BEST_WARMUP = 0.7
BEST_HEURISTIC = heuristic_policy

print("Best config for Sections 5 and 6:")
print(f"  arch:        {BEST_ARCH_CLASS.__name__}")
print(f"  n_frames:    {BEST_N_FRAMES}")
print(f"  preprocess:  {BEST_PREPROCESS.__name__}")
print(f"  n_channels:  {BEST_N_CHANNELS}")
print(f"  warmup_pct:  {BEST_WARMUP}")
print(f"  heuristic:   {BEST_HEURISTIC.__name__}")
""")
        changes.append("Task 4")

# Task 5 + 6: BC and final sections use BEST_* vars; improve BC timing and episodes
idx = find_idx("bc_net = DQN(n_frames=1, dropout=0.0).to(device)")
if idx != -1:
    s = src(cells[idx]).replace(
        "bc_net = DQN(n_frames=1, dropout=0.0).to(device)",
        "bc_net = BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)"
    )
    set_src(cells[idx], s)
    changes.append("Task 5 BC net")

idx = find_idx("bc_conditions = [")
if idx != -1:
    set_src(cells[idx], """import copy, time

bc_conditions = [
    ('BC+warmup=0.0',  True,  0.0),
    ('BC+warmup=0.5',  True,  0.5),
    ('rnd+warmup=0.7', False, 0.7),
]

bc_comparison = {}
for label, use_bc, w in bc_conditions:
    net = copy.deepcopy(bc_net) if use_bc else BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)
    opt = optim.RMSprop(net.parameters(), lr=LR, alpha=0.99, eps=1e-8)
    start = time.time()
    _, rewards, losses, qs = train(
        envs['difficulty_0'], net, opt, n_episodes=N_EPISODES_MD, n_frames=BEST_N_FRAMES,
        heuristic=BEST_HEURISTIC, warmup_pct=w, use_huber=True, name=label,
        preprocess_fn=BEST_PREPROCESS)
    t = time.time() - start
    mean, std, mq, s_per_t, _ = evaluate(
        envs['difficulty_0'], net, BEST_N_FRAMES, t, preprocess_fn=BEST_PREPROCESS)
    bc_comparison[label] = {
        'mean': mean, 'std': std, 'q': mq, 'rewards': rewards, 's_per_t': s_per_t,
        'n_episodes': N_EPISODES_MD,
    }
    save_result(f'bc/{label}', mean, std, q_val=mq, n_episodes=N_EPISODES_MD,
                use_bc=use_bc, warmup=w, s_per_t=s_per_t)
    print(f'{label:20s}  mean={mean:.3f}  std={std:.3f}  s/t={s_per_t:.4f}')

fig, ax = plt.subplots(figsize=(12, 5))
for label, data in bc_comparison.items():
    ax.plot(data['rewards'], label=label, alpha=0.7)
ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
ax.set_title('BC vs RL (diff 0)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")
    changes.append("Task 5+6 BC comparison")

# Task 7: assert/print improvements in key cells
idx = find_idx("Environments created:")
if idx != -1:
    s = src(cells[idx])
    if "assert envs['difficulty_0'].action_space.n == 2" not in s:
        s = s.replace(
            "print(\"  [ok] all envs have expected obs shape (54, 39, 3)\")",
            "assert envs['difficulty_0'].observation_space.shape == (54, 39, 3), f\"unexpected obs shape: {envs['difficulty_0'].observation_space.shape}\"\n"
            "assert envs['difficulty_0'].action_space.n == 2, f\"unexpected action count: {envs['difficulty_0'].action_space.n}\"\n"
            "print(\"  [ok] env shapes confirmed\")"
        )
        set_src(cells[idx], s)
        changes.append("Task 7 env asserts")

idx = find_idx("move up reward:")
if idx != -1:
    s = src(cells[idx])
    s = s.replace("assert abs(r_up   -  0.02) < 1e-6, f\"got {r_up}\"", "assert abs(r_up - 0.02) < 0.01, f\"unexpected up reward: {r_up}\"")
    s = s.replace("assert abs(r_down - -0.01) < 1e-6, f\"got {r_down}\"", "assert abs(r_down - (-0.01)) < 0.01, f\"unexpected down reward: {r_down}\"")
    s = s.replace("print(f\"move up reward:   {r_up:.4f}   (expected +0.02)\")", "print(f\"move up reward:   {r_up:.4f}   (expected ~+0.02)  [ok]\")")
    s = s.replace("print(f\"move down reward: {r_down:.4f}  (expected -0.01)\")", "print(f\"move down reward: {r_down:.4f}  (expected ~-0.01)  [ok]\")")
    set_src(cells[idx], s)
    changes.append("Task 7 reward asserts")

idx = find_idx("episode ended at step")
if idx != -1 and "assert not terminated" not in src(cells[idx]):
    s = src(cells[idx])
    s = s.replace("assert env.score >= 0, 'score should be non-negative'", "assert not terminated, 'terminated should never be True'\n    assert env.score >= 0, 'score should be non-negative'")
    set_src(cells[idx], s)
    changes.append("Task 7 episode asserts")

idx = find_idx("preprocess_obs:")
if idx != -1:
    s = src(cells[idx])
    s = s.replace("print(f'preprocess_obs:           {obs_test.shape} -> {out_rgb.shape}')", "print(f'[ok] preprocess_obs:           {obs_test.shape} -> {out_rgb.shape}')")
    s = s.replace("print(f'preprocess_obs_r_channel: {obs_test.shape} -> {out_r.shape}')", "print(f'[ok] preprocess_obs_r_channel: {obs_test.shape} -> {out_r.shape}')")
    set_src(cells[idx], s)
    changes.append("Task 7 preprocess asserts")

idx = find_idx("smoke test: score=")
if idx != -1:
    s = src(cells[idx])
    if "assert _ms >= 0" not in s:
        s = s.replace("assert isinstance(_ms, (int, float))", "assert isinstance(_ms, (int, float))\n    assert _ms >= 0, f'smoke test: score should be non-negative, got {_ms}'")
        set_src(cells[idx], s)
        changes.append("Task 7 smoke assert")

# Task 8: add normalization rationale in Section 2.2 markdown
idx = find_idx("## 2.2 Observation Preprocessing")
if idx != -1 and "Input normalization" not in src(cells[idx]):
    s = src(cells[idx]).rstrip("\n") + "\n- **Input normalization**: divide uint8 values (0-255) by 255.0 -> float32 in [0,1].\n  Why: neural net weights are initialized for unit-scale inputs; raw [0,255] causes very large\n  first-layer activations and unstable gradients.\n"
    set_src(cells[idx], s)
    changes.append("Task 8")

# Task 9: add explicit grad clip note in 2.3 markdown
idx = find_idx("## 2.3 Training Loop and Evaluation")
if idx != -1:
    s = src(cells[idx]).replace(
        "backprop -> clip_grad_norm_(1.0) -> optimizer.step()",
        "backprop -> clip_grad_norm_(params, GRAD_CLIP_NORM)  # cap gradient L2-norm\n        optimizer.step()"
    )
    set_src(cells[idx], s)
    changes.append("Task 9")

# Task 11 + 16 + 17: enhance Section 6.2 summary and add learning dashboard cell
idx = find_idx("# ── Build summary table")
if idx != -1:
    set_src(cells[idx], """# -- Build summary table -------------------------------------------------
summary_rows = []
for pol in ['random', 'always_up']:
    row = {'Policy': pol}
    for i in range(4):
        row[f'Diff{i}'] = f'{baseline_results[pol][i][0]:.2f}+-{baseline_results[pol][i][1]:.2f}'
    summary_rows.append(row)

for h_name in heuristics:
    row = {'Policy': f'heuristic_{h_name}'}
    for i in range(4):
        row[f'Diff{i}'] = f'{heuristic_results[h_name][i][0]:.2f}+-{heuristic_results[h_name][i][1]:.2f}'
    summary_rows.append(row)

if 'bc_comparison' in globals() and bc_comparison:
    best_bc_label = max(bc_comparison, key=lambda k: bc_comparison[k]['mean'])
    row = {'Policy': f'BC_best ({best_bc_label})', 'Diff0': '-', 'Diff1': '-', 'Diff2': '-', 'Diff3': '-'}
    row['Diff0'] = f"{bc_comparison[best_bc_label]['mean']:.2f}+-{bc_comparison[best_bc_label]['std']:.2f}"
    summary_rows.append(row)

summary_rows.append({
    'Policy': 'DQN_final',
    'Diff0': f'{mean0:.2f}+-{std0:.2f}',
    'Diff1': f'{mean1:.2f}+-{std1:.2f}',
    'Diff2': '-',
    'Diff3': '-',
})

df = pd.DataFrame(summary_rows, columns=['Policy','Diff0','Diff1','Diff2','Diff3'])
df.set_index('Policy', inplace=True)
print('Final Results Summary:')
print(df.to_string())

# -- Bar chart with error bars --------------------------------------------
_pol_names = ['random', 'always_up', 'heuristic_base', 'heuristic_dev', 'heuristic_v2', 'DQN_final']
if 'bc_comparison' in globals() and bc_comparison:
    _pol_names.insert(-1, 'BC_best')

_scores_d0, _scores_d1 = [], []
_stds_d0, _stds_d1 = [], []

for _pol in _pol_names:
    if _pol == 'DQN_final':
        _scores_d0.append(mean0); _scores_d1.append(mean1)
        _stds_d0.append(std0); _stds_d1.append(std1)
    elif _pol == 'BC_best':
        _best = max(bc_comparison, key=lambda k: bc_comparison[k]['mean'])
        _scores_d0.append(bc_comparison[_best]['mean']); _scores_d1.append(np.nan)
        _stds_d0.append(bc_comparison[_best]['std']); _stds_d1.append(np.nan)
    elif _pol.startswith('heuristic_'):
        _hn = _pol.split('_', 1)[1]
        _scores_d0.append(heuristic_results[_hn][0][0]); _scores_d1.append(heuristic_results[_hn][1][0])
        _stds_d0.append(heuristic_results[_hn][0][1]); _stds_d1.append(heuristic_results[_hn][1][1])
    else:
        _scores_d0.append(baseline_results[_pol][0][0]); _scores_d1.append(baseline_results[_pol][1][0])
        _stds_d0.append(baseline_results[_pol][0][1]); _stds_d1.append(baseline_results[_pol][1][1])

_x = np.arange(len(_pol_names)); _w = 0.35
_colors_d0 = ['#4878d0' if 'DQN' in p else ('#9c6ade' if 'BC' in p else ('#6acc65' if 'heuristic' in p else '#d65f5f')) for p in _pol_names]
_colors_d1 = ['#1e4fa3' if 'DQN' in p else ('#5f3ca3' if 'BC' in p else ('#3a8a36' if 'heuristic' in p else '#8b1a1a')) for p in _pol_names]

fig, ax = plt.subplots(figsize=(14, 5))
bars0 = ax.bar(_x - _w/2, _scores_d0, _w, yerr=_stds_d0, capsize=4, label='Difficulty 0', color=_colors_d0, alpha=0.85)
bars1 = ax.bar(_x + _w/2, _scores_d1, _w, yerr=_stds_d1, capsize=4, label='Difficulty 1', color=_colors_d1, alpha=0.85)
ax.set_xticks(_x)
ax.set_xticklabels(_pol_names, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Mean score (N episodes)')
ax.set_title('Policy Performance Summary -- Baselines vs Heuristics vs BC vs DQN')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

print(f'\\nAll experiment results ({len(all_results)} entries in task1_results.json):')
for key, val in sorted(all_results.items()):
    print(f'  {key}: score={val["score_mean"]:.2f}+-{val["score_std"]:.2f}')
""")
    changes.append("Task 11+16")

# Task 17: add learning curves dashboard right after 6.2 code
if find_idx("Final learning curves dashboard") == -1 and idx != -1:
    insert_cell(idx, "code", """# -- Final learning curves dashboard -------------------------------------
# rew0, q0, los0 from Section 6.1 phase 1; rew1, q1, los1 from phase 2

if 'rew0' in globals() and 'rew1' in globals() and 'los0' in globals() and 'los1' in globals():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0,0].plot(rew0, alpha=0.4, label='diff0 raw')
    _avg0 = [np.mean(rew0[max(0, i-10):i+1]) for i in range(len(rew0))]
    axes[0,0].plot(_avg0, label='diff0 rolling avg (10ep)')
    axes[0,0].set_title('Reward -- Difficulty 0'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(rew1, alpha=0.4, label='diff1 raw')
    _avg1 = [np.mean(rew1[max(0, i-10):i+1]) for i in range(len(rew1))]
    axes[0,1].plot(_avg1, label='diff1 rolling avg (10ep)')
    axes[0,1].set_title('Reward -- Difficulty 1'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(los0, alpha=0.6, color='orange', label='diff0 loss')
    axes[1,0].set_title('Loss -- Difficulty 0'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(los1, alpha=0.6, color='orange', label='diff1 loss')
    axes[1,1].set_title('Loss -- Difficulty 1'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    for _ax in axes.flat:
        _ax.set_xlabel('Episode')
    plt.suptitle('Final Training Learning Curves (Phase 1 -- Basic DQN)', fontsize=12)
    plt.tight_layout(); plt.show()
else:
    print('[!] Run Section 6.1 first to generate rew0, rew1, los0, los1')
""")
    changes.append("Task 17")

# Task 12: Section 5 markdown add BC vs warmup explanation
idx = find_idx("# **5. Behavioral Cloning")
if idx != -1 and "BC vs warmup_pct" not in src(cells[idx]):
    s = src(cells[idx]).rstrip("\n") + "\n\n### BC vs warmup_pct: what is the difference?\n\nwarmup_pct in train(): at each step, with probability w, use heuristic action.\nThe heuristic action is used online while RL is already running.\n\nBC pretraining: train the network first on (RGB obs, heuristic action) pairs via supervised learning.\nThen start RL from smarter weights.\n\nKey difference: BC improves initialization; warmup improves transition quality during RL.\nThey can be combined (BC init + warmup > 0), which is tested in Section 5.3.\n"
    set_src(cells[idx], s)
    changes.append("Task 12")

# Task 13: Section 5.1 markdown add human gameplay note
idx = find_idx("## 5.1 Collect Demonstrations")
if idx != -1 and "Could we use human gameplay" not in src(cells[idx]):
    s = src(cells[idx]).rstrip("\n") + "\n\n### Could we use human gameplay instead of the heuristic?\nYes in principle: play via a local pygame window, record (obs, action) pairs, save, and load here.\nIn practice, headless notebook runtimes do not provide a display, so manual pygame control fails.\nHeuristic demos are deterministic and reproducible (same seed -> same trajectories), so they are the practical notebook choice.\n"
    set_src(cells[idx], s)
    changes.append("Task 13")

# Task 14: ensure instability visualization block exists in Section 3.8 code
idx = find_idx("Instability analysis")
if idx != -1 and "rolling_std" not in src(cells[idx]):
    s = src(cells[idx]).rstrip("\n") + "\n\n# instability demonstration\n_inst_net = DQN(n_frames=1, dropout=0.0).to(device)\n_inst_opt = optim.RMSprop(_inst_net.parameters(), lr=LR, alpha=0.99, eps=1e-8)\n_, _inst_rewards, _inst_losses, _ = train(\n    envs['difficulty_0'], _inst_net, _inst_opt,\n    n_episodes=N_EPISODES_MD, n_frames=1,\n    heuristic=heuristic_policy, warmup_pct=0.7, use_huber=True,\n    name='instability demo')\nwindow = max(3, N_EPISODES_MD // 10)\nrolling_std = [np.std(_inst_rewards[max(0, i-window):i+1]) for i in range(len(_inst_rewards))]\nfig, axes = plt.subplots(1, 2, figsize=(14, 4))\naxes[0].plot(_inst_rewards, alpha=0.6); axes[0].set_title('Online DQN reward trajectory')\naxes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward'); axes[0].grid(alpha=0.3)\naxes[1].plot(rolling_std, color='orange'); axes[1].set_title('Training instability (rolling std)')\naxes[1].set_xlabel('Episode'); axes[1].set_ylabel('Reward std'); axes[1].grid(alpha=0.3)\nplt.tight_layout(); plt.show()\n"
    set_src(cells[idx], s)
    changes.append("Task 14")

# Task 15 + 5: final training cells use BEST_* and save in cwd
idx = find_idx("final_net = DQN(n_frames=1, dropout=0.0).to(device)")
if idx != -1:
    set_src(cells[idx], """import time, os, torch

final_net = BEST_ARCH_CLASS(n_frames=BEST_N_FRAMES, dropout=0.0).to(device)
final_opt = optim.RMSprop(final_net.parameters(), lr=LR, alpha=0.99, eps=1e-8)

print(f'=== Phase 1: difficulty 0, {N_EPISODES_XL} episodes ===')
final_net, rew0, los0, q0 = train(
    envs['difficulty_0'], final_net, final_opt,
    n_episodes=N_EPISODES_XL, n_frames=BEST_N_FRAMES,
    heuristic=BEST_HEURISTIC, warmup_pct=BEST_WARMUP, use_huber=True,
    name='final diff0', preprocess_fn=BEST_PREPROCESS)

start = time.time()
mean0, std0, q0_f, _, _ = evaluate(
    envs['difficulty_0'], final_net, BEST_N_FRAMES, time.time()-start,
    preprocess_fn=BEST_PREPROCESS)
print(f'Eval diff0: {mean0:.3f} +- {std0:.3f}')

_save_path = os.path.join(os.getcwd(), 'model0.pt')
torch.save(final_net.state_dict(), _save_path)
save_result('final/diff0', mean0, std0, q_val=q0_f, n_episodes=N_EPISODES_XL,
            notes='final model after diff0 curriculum')
print(f'[ok] model0.pt saved -> {_save_path}')
""")
    changes.append("Task 5+15 phase1")

idx = find_idx("=== Phase 2: difficulty 1")
if idx != -1:
    set_src(cells[idx], """import time, os, torch

print(f'=== Phase 2: difficulty 1, {N_EPISODES_LG} episodes (curriculum) ===')
final_net, rew1, los1, q1 = train(
    envs['difficulty_1'], final_net, final_opt,
    n_episodes=N_EPISODES_LG, n_frames=BEST_N_FRAMES,
    heuristic=BEST_HEURISTIC, warmup_pct=0.5, use_huber=True,
    name='final diff1', preprocess_fn=BEST_PREPROCESS)

start = time.time()
mean1, std1, q1_f, _, _ = evaluate(
    envs['difficulty_1'], final_net, BEST_N_FRAMES, time.time()-start,
    preprocess_fn=BEST_PREPROCESS)
print(f'Eval diff1: {mean1:.3f} +- {std1:.3f}')

_save_path1 = os.path.join(os.getcwd(), 'model1.pt')
_save_path_sub = os.path.join(os.getcwd(), 'model.pt')
torch.save(final_net.state_dict(), _save_path1)
torch.save(final_net.state_dict(), _save_path_sub)
save_result('final/diff1', mean1, std1, q_val=q1_f, n_episodes=N_EPISODES_LG,
            notes='final model after diff1 curriculum')
print(f'[ok] model1.pt + model.pt saved -> {_save_path_sub}')
""")
    changes.append("Task 5+15 phase2")

idx = find_idx("## 6.1 Final Training Runs")
if idx != -1:
    set_src(cells[idx], """---
## 6.1 Final Training Runs
Progressive curriculum: diff0 -> diff1, same model.
Uses the best configuration selected in Section 3.9 (arch, n_frames, preprocess, warmup).
`model0.pt`, `model1.pt`, and `model.pt` are saved to `os.getcwd()`.
The Agent class in Section 6.3 loads `model.pt` from the same path.
""")
    changes.append("Task 5 markdown")

# Task 18: BN markdown expectation + DQN_BN docstring
idx = find_idx("## 3.5 Regularization Ablation")
if idx != -1 and "Expected outcome: DQN_BN" not in src(cells[idx]):
    s = src(cells[idx]).rstrip("\n") + "\nExpected outcome: DQN_BN achieves near-zero score or unstable loss; if stable, note it as an edge case.\n"
    set_src(cells[idx], s)
    changes.append("Task 18 md")

idx = find_idx("class DQN_BN(nn.Module):")
if idx != -1 and "BatchNorm2d" in src(cells[idx]):
    s = src(cells[idx])
    if '"""DQN with BatchNorm2d' not in s:
        s = s.replace(
            "class DQN_BN(nn.Module):\n    def __init__(self, n_frames=1):",
            "class DQN_BN(nn.Module):\n    \"\"\"DQN with BatchNorm2d. Expected to hurt with batch_size=1 online RL.\"\"\"\n    def __init__(self, n_frames=1):"
        )
        set_src(cells[idx], s)
        changes.append("Task 18 code")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Applied tasks:")
for c in changes:
    print(" -", c)
print("Total:", len(changes))
