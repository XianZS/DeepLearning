import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans

# 设置全局字体和风格
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

# ===================== 创建画布：2个子图对应两种极端场景 =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# =============== 极端场景1：初始质心极差 → 陷入局部最优 ===============
# 生成4个簇的标准数据
X1, y_true1 = make_blobs(n_samples=400, centers=4, cluster_std=0.7, random_state=10)

# 手动指定极差的初始质心（3个质心都挤在右侧同一个簇附近）
bad_init = np.array(
    [
        [6.5, -8],  # 都集中在右下角簇内
        [7.2, -7.5],
        [7.0, -8.5],
        [1, -6],  # 只有1个质心在左侧
    ]
)

# 用坏初始值运行K-Means
kmeans_bad = KMeans(n_clusters=4, init=bad_init, n_init=1, max_iter=100, random_state=0)
labels_bad = kmeans_bad.fit_predict(X1)
centroids_bad = kmeans_bad.cluster_centers_

# 绘制聚类结果
for i in range(4):
    ax1.scatter(
        X1[labels_bad == i, 0],
        X1[labels_bad == i, 1],
        c=colors[i],
        s=40,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )
ax1.scatter(
    centroids_bad[:, 0],
    centroids_bad[:, 1],
    s=250,
    c="red",
    marker="X",
    edgecolors="black",
    linewidth=2,
    zorder=5,
)

# --- 添加注解 ---
# 注解1：指出质心扎堆
ax1.annotate(
    "初始质心全部扎堆在此，\n导致单个簇被强行拆分",
    xy=(7, -8),
    xytext=(8.5, -6),
    fontsize=11,
    arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=8),
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="red", pad=4),
)

# 注解2：指出簇被错误合并
ax1.annotate(
    "左侧两个真实簇\n被错误合并为一个",
    xy=(1, -5),
    xytext=(-3, -2),
    fontsize=11,
    arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=8),
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="red", pad=4),
)

ax1.set_title("极端场景1：初始质心极差 → 局部最优解", fontsize=13)
ax1.grid(True, alpha=0.3)

# =============== 极端场景2：非凸半月形数据 → 原理性失效 ===============
# 生成半月形数据
X2, y_true2 = make_moons(n_samples=300, noise=0.08, random_state=42)

# 运行K-Means (k=2)
kmeans_moon = KMeans(n_clusters=2, n_init=10, random_state=0)
labels_moon = kmeans_moon.fit_predict(X2)
centroids_moon = kmeans_moon.cluster_centers_

# 绘制聚类结果
for i in range(2):
    ax2.scatter(
        X2[labels_moon == i, 0],
        X2[labels_moon == i, 1],
        c=colors[i],
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )
ax2.scatter(
    centroids_moon[:, 0],
    centroids_moon[:, 1],
    s=250,
    c="red",
    marker="X",
    edgecolors="black",
    linewidth=2,
    zorder=5,
)

# --- 添加注解 ---
ax2.annotate(
    "同一条月牙被\n垂直切分成两个簇",
    xy=(0.5, 0.25),
    xytext=(0.6, 0.8),
    fontsize=11,
    arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=8),
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="red", pad=4),
)

ax2.annotate(
    "K-Means本质是球形划分\n无法识别非凸的月牙结构",
    xy=(-0.2, 0.6),
    xytext=(-1.2, 0.9),
    fontsize=11,
    arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=8),
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="red", pad=4),
)

ax2.set_title("极端场景2：非凸半月形数据 → 原理性失效", fontsize=13)
ax2.grid(True, alpha=0.3)
plt.savefig("./test.png")
plt.tight_layout()
plt.show()
