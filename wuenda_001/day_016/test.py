import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 环境参数 --------------------------
gamma = 0.5
step_reward = 2
terminal_left_reward = 100
terminal_right_reward = 10
s_example = 3
num_states = 6


# -------------------------- 2. 计算累加层级 --------------------------
def get_cumulative_layers(s, direction):
    layers = []
    current_s = s
    if direction == "left":
        while current_s > 0:
            layers.append(step_reward * (gamma ** len(layers)))
            current_s -= 1
        layers.append(terminal_left_reward * (gamma ** len(layers)))
    else:
        while current_s < num_states - 1:
            layers.append(step_reward * (gamma ** len(layers)))
            current_s += 1
        layers.append(terminal_right_reward * (gamma ** len(layers)))
    return np.array(layers)


layers_left = get_cumulative_layers(s_example, "left")
layers_right = get_cumulative_layers(s_example, "right")
q_left = layers_left.sum()
q_right = layers_right.sum()

all_q_left = np.zeros(num_states)
all_q_right = np.zeros(num_states)
for s in range(num_states):
    all_q_left[s] = get_cumulative_layers(s, "left").sum()
    all_q_right[s] = get_cumulative_layers(s, "right").sum()

# -------------------------- 3. 绘图 --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), dpi=120, gridspec_kw={"width_ratios": [1.2, 1]}
)

bar_positions = [0, 1.5]
bar_width = 0.6
colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]

# 向左动作堆叠条
bottom = 0
for i, val in enumerate(layers_left):
    ax1.bar(
        bar_positions[0],
        val,
        width=bar_width,
        bottom=bottom,
        color=colors[i % len(colors)],
        edgecolor="white",
        linewidth=1.2,
    )
    label = (
        f"$\gamma^{i} \\times R_{i + 1} = {val:.2f}$"
        if i < len(layers_left) - 1
        else f"$\gamma^{i} \\times R_{{终端}} = {val:.2f}$"
    )
    ax1.text(
        bar_positions[0],
        bottom + val / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    bottom += val
ax1.text(
    bar_positions[0],
    bottom + 0.8,
    f"Q(状态{s_example}, 向左) = {q_left:.2f}",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="#1e40af",
)

# 向右动作堆叠条
bottom = 0
for i, val in enumerate(layers_right):
    ax1.bar(
        bar_positions[1],
        val,
        width=bar_width,
        bottom=bottom,
        color=colors[i % len(colors)],
        edgecolor="white",
        linewidth=1.2,
    )
    label = (
        f"$\gamma^{i} \\times R_{i + 1} = {val:.2f}$"
        if i < len(layers_right) - 1
        else f"$\gamma^{i} \\times R_{{终端}} = {val:.2f}$"
    )
    ax1.text(
        bar_positions[1],
        bottom + val / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    bottom += val
ax1.text(
    bar_positions[1],
    bottom + 0.8,
    f"Q(状态{s_example}, 向右) = {q_right:.2f}",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="#1e40af",
)

ax1.set_xticks(bar_positions)
ax1.set_xticklabels(["向左动作", "向右动作"], fontsize=11)
ax1.set_title(
    f"Q值的累加递进层级（状态{s_example}，折扣率 γ = {gamma}）", fontsize=13, pad=15
)
ax1.set_ylabel("折现奖励大小", fontsize=11)
ax1.set_ylim(0, max(q_left, q_right) * 1.15)
ax1.grid(axis="y", alpha=0.3)

# 子图2：所有状态Q值对比
x = np.arange(num_states)
ax2.plot(
    x, all_q_left, marker="o", linewidth=2, color="#2563eb", label="向左动作 Q(s,左)"
)
ax2.plot(
    x, all_q_right, marker="s", linewidth=2, color="#f59e0b", label="向右动作 Q(s,右)"
)

for s in range(num_states):
    dist_l = s
    dist_r = num_states - 1 - s
    ax2.text(
        s,
        all_q_left[s] + 0.8,
        f"距左终点{dist_l}步",
        ha="center",
        fontsize=8,
        color="#1e40af",
    )
    ax2.text(
        s,
        all_q_right[s] - 1.2,
        f"距右终点{dist_r}步",
        ha="center",
        fontsize=8,
        color="#b45309",
    )

ax2.set_title("所有状态的Q值与距离的关系", fontsize=13, pad=15)
ax2.set_xlabel("状态编号", fontsize=11)
ax2.set_ylabel("Q(s,a) 总回报", fontsize=11)
ax2.set_xticks(x)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
