"""
generate_graphs.py
Generates publication-quality benchmark plots comparing Base MiniCPM5-1B, Distilled MiniCPM5-1B, and Gemma-4-31B Teacher.
Saves PNG charts locally and in artifacts directory for visual proof.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

OUTPUT_DIR = "C:/Users/arro/.gemini/antigravity-cli/brain/2d97621b-286a-4703-b84f-5caea6c98d16/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Chart 1: Inference Throughput (tokens/sec) vs VRAM (GB)
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))

models = ['Base MiniCPM5-1B', 'Distilled MiniCPM5-1B\n(Ours)', 'Gemma-4-31B\nTeacher']
throughput = [62.00, 76.27, 12.40]
vram = [2.40, 2.40, 38.50]

color1 = '#2b5c8f'
color2 = '#d95f02'

x = range(len(models))
width = 0.35

rects1 = ax1.bar([p - width/2 for p in x], throughput, width, label='Throughput (tok/s)', color=color1)
ax1.set_ylabel('Inference Throughput (tokens/sec)', color=color1, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
rects2 = ax2.bar([p + width/2 for p in x], vram, width, label='VRAM Footprint (GB)', color=color2)
ax2.set_ylabel('Peak VRAM Memory (GB)', color=color2, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

plt.xticks(x, models, fontweight='bold')
plt.title('Figure 1: Inference Throughput vs VRAM Memory Footprint', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()

chart1_path = os.path.join(OUTPUT_DIR, "throughput_vs_vram.png")
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"[+] Saved Chart 1 to {chart1_path}")

# ---------------------------------------------------------
# Chart 2: Zero-Shot JSON Accuracy Across 3 Iterations (%)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

iterations = ['Base Model', 'Iteration 1\n(Chat Tokens)', 'Iteration 2\n(Aligned WINNER)', 'Iteration 3\n(Multi-Domain)']
accuracy = [34.2, 0.0, 70.0, 0.0]
colors = ['#7570b3', '#e7298a', '#1b9e77', '#d95f02']

bars = ax.bar(iterations, accuracy, color=colors, width=0.5)
ax.set_ylabel('Out-of-Domain JSON Accuracy (%)', fontweight='bold')
ax.set_ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.title('Figure 2: Zero-Shot JSON Accuracy on Real-World Benchmark (suneeldk/text-json)', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()

chart2_path = os.path.join(OUTPUT_DIR, "accuracy_across_iterations.png")
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"[+] Saved Chart 2 to {chart2_path}")

# ---------------------------------------------------------
# Chart 3: Loss Convergence Trajectory
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

epochs = [1, 2, 3]
baseline_loss = [17279.31, 15839.34, 14536.51]
gemma31b_loss = [18083.09, 15527.52, 14546.81]
iter2_loss = [8721.13, 6962.28, 6612.65]

ax.plot(epochs, baseline_loss, marker='o', linewidth=2.5, label='Gemma-4-E4B Teacher Distillation', color='#2b5c8f')
ax.plot(epochs, gemma31b_loss, marker='s', linewidth=2.5, label='Gemma-4-31B Teacher Distillation', color='#d95f02')
ax.plot(epochs, iter2_loss, marker='^', linewidth=2.5, label='Iteration 2 Aligned Distillation (WINNER)', color='#1b9e77')

ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Distillation Loss', fontweight='bold')
ax.set_xticks(epochs)
plt.title('Figure 3: Training Loss Convergence Trajectory Across Epochs', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right')
fig.tight_layout()

chart3_path = os.path.join(OUTPUT_DIR, "loss_convergence.png")
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"[+] Saved Chart 3 to {chart3_path}")

print("[+] All 3 visual proof benchmark graphs generated successfully!")
