#!/usr/bin/env python
"""Generate GRPO training curve figure from TensorBoard events."""
import sys
sys.path.insert(0, '/home/v-zezhouwang/hydrogen-patent-research')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TB_DIR = '4_GRPO/tensorboard_log/hydrogen_patent_grpo_online/qwen25_7b_hydrogen_grpo_online_4gpu'
OUT_PATH = 'dpmlretriever/paper/figures/training_curve.pdf'

ea = EventAccumulator(TB_DIR)
ea.Reload()

# Extract metrics
def get_scalar(tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)

reward_steps, reward_vals = get_scalar('critic/rewards/mean')
val_steps, val_vals = get_scalar('val-core/hydrogen_patent_cypher_online/reward/mean@1')
kl_steps, kl_vals = get_scalar('actor/kl_loss')

# Plot
fig, ax1 = plt.subplots(figsize=(8, 4.5))

color_reward = '#2196F3'
color_val = '#4CAF50'
color_kl = '#FF5722'

# Left y-axis: reward and validation score
ax1.plot(reward_steps, reward_vals, color=color_reward, alpha=0.4, linewidth=0.8)
# Smoothed reward
window = 10
if len(reward_vals) > window:
    smoothed = np.convolve(reward_vals, np.ones(window)/window, mode='valid')
    ax1.plot(reward_steps[window-1:], smoothed, color=color_reward, linewidth=2, label='Training Reward (smoothed)')
ax1.scatter(val_steps, val_vals, color=color_val, s=50, zorder=5, label='Validation Score (mean@1)', marker='D')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Reward / Score', fontsize=12, color='black')
ax1.set_ylim(-0.1, 1.0)
ax1.tick_params(axis='y')

# Right y-axis: KL divergence
ax2 = ax1.twinx()
ax2.plot(kl_steps, kl_vals, color=color_kl, linewidth=1.5, linestyle='--', alpha=0.7, label='KL Divergence')
ax2.set_ylabel('KL Divergence', fontsize=12, color=color_kl)
ax2.tick_params(axis='y', labelcolor=color_kl)
ax2.set_ylim(-0.005, 0.06)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9, framealpha=0.9)

ax1.set_title('GRPO Training Dynamics (Qwen2.5-7B, 4×A100)', fontsize=13)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')

# Print key numbers for verification
print(f'\nReward: {reward_vals[0]:.4f} (step {reward_steps[0]}) -> {reward_vals[-1]:.4f} (step {reward_steps[-1]})')
print(f'Validation: {val_vals[0]:.4f} (step {val_steps[0]}) -> {val_vals[-1]:.4f} (step {val_steps[-1]})')
print(f'KL: {kl_vals[0]:.4f} (step {kl_steps[0]}) -> {kl_vals[-1]:.4f} (step {kl_steps[-1]})')
