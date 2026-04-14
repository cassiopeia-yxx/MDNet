import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====== 参数 ======
img_path = r"F:\BaiduNetdiskDownload\LOLBlur\test\low_blur_noise\0017\0084.png"   # 改成你的图像路径
lam = 10.0
sigma = 10.0

# ====== 读取图像 ======
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 转灰度并归一化到 [0, 255] 浮点
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

# ====== Sobel梯度 ======
grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

# 梯度幅值
grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

# ====== 计算 K 和 G ======
K = 1.0 + lam * np.exp(-grad_mag / sigma)

Gx = K * grad_x
Gy = K * grad_y
G_mag = K * grad_mag

# ====== 可视化归一化 ======
def norm_to_01(x):
    x = x.astype(np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

grad_mag_vis = norm_to_01(grad_mag)
K_vis = norm_to_01(K)
G_mag_vis = norm_to_01(G_mag)

# ====== 显示 ======
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Input")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(grad_mag_vis, cmap="gray")
plt.title("|∇I|")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(K_vis, cmap="jet")
plt.title("K")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(G_mag_vis, cmap="gray")
plt.title("|G| = K·|∇I|")
plt.axis("off")

plt.tight_layout()
plt.show()