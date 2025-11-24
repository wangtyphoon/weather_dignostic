import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# 設定圖片路徑 (請依據你的檔案名稱修改)
# 注意層次順序：地面在最下，500hPa在最上XX
img_paths = {
    '500hPa': '500hpa.png',  # 最上層
    '850hPa': '850hpa.png',  # 中間層
    'Surface': '1000hpa.png'  # 最下層
}

def load_image(path):
    # 讀取圖片並轉為 RGB 陣列，歸一化到 0-1 之間
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0

def crop_white_borders(images, threshold=0.98):
    """
    移除所有圖共同的白色邊框，保留至少有內容的部分。
    threshold 越低，越容易被判定為內容。
    """
    first = next(iter(images.values()))
    mask = np.zeros(first.shape[:2], dtype=bool)

    # 合併所有圖的非白色區域
    for img in images.values():
        gray = img.mean(axis=2)
        mask |= gray < threshold

    if not mask.any():
        return images  # 全白圖，直接返回

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cropped = {}
    for name, img in images.items():
        cropped[name] = img[y_min : y_max + 1, x_min : x_max + 1]
    return cropped

fig = plt.figure(figsize=(10, 10), dpi=400)
ax = fig.add_subplot(111, projection='3d')

# 設定各層的高度 (Z軸位置)
z_levels = {'Surface': 0, '850hPa': 4, '500hPa': 8}
alphas = {'Surface': 1.0, '850hPa': 0.6, '500hPa': 0.5}  # 設定透明度，越上層越透明

# 讀取所有圖片並裁切共同白邊，保持對齊
raw_images = {name: load_image(path) for name, path in img_paths.items()}
images = crop_white_borders(raw_images, threshold=0.98)

# 為了讓圖顯示正確，我們需要建立網格 (使用像素索引避免重新取樣造成模糊)
sample_img = next(iter(images.values()))
ny, nx, _ = sample_img.shape
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
Y = np.flipud(Y)  # 或者 ax.set_ylim(ax.get_ylim()[::-1])

# 繪圖迴圈
for level_name, path in img_paths.items():
    img_data = images[level_name]
    print(f"Loaded {level_name} image with shape: {img_data.shape}")
    
    # Z 軸高度
    Z = np.ones(X.shape) * z_levels[level_name]
    
    # 使用 plot_surface，並將圖片貼圖(facecolors)設為該圖片
    # rstride 和 cstride 設大一點可以加速渲染，設為 1 畫質最好但較慢
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=img_data,
        rstride=1,  # 最細網格貼圖，避免模糊
        cstride=1,
        shade=False,
        alpha=alphas[level_name],
        antialiased=False  # 關閉抗鋸齒以保留原圖清晰度
    )

# 調整視角 (Elevation: 仰角, Azimuth: 方位角)
ax.view_init(elev=20, azim=-60)

# 隱藏座標軸，讓畫面更乾淨
ax.set_axis_off()

plt.tight_layout()
plt.show()
