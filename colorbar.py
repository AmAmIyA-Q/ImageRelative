import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_image_chinese_path(image_path):
    """支持中文路径的图像读取"""
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def extract_palette(image_path, n_colors=5, resize_width=200):
    # 使用 imdecode 读取图像（支持中文路径）
    img = read_image_chinese_path(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 压缩图像尺寸
    h, w = img.shape[:2]
    new_h = int(h * resize_width / w)
    img = cv2.resize(img, (resize_width, new_h))
    
    # 转换为二维数组并聚类
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    
    # 按出现频率排序
    sorted_idx = np.argsort(-counts)
    colors = colors[sorted_idx]
    counts = counts[sorted_idx]

    # 生成颜色条图像
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    start = 0
    for i, (color, count) in enumerate(zip(colors, counts)):
        ratio = count / sum(counts)
        end = start + int(ratio * palette.shape[1])
        palette[:, start:end, :] = color
        start = end

    return colors, palette

def show_palette(image_path):
    colors, palette = extract_palette(image_path)
    plt.figure(figsize=(6, 2))
    plt.imshow(palette)
    plt.axis('off')
    plt.title('Dominant Colors')
    plt.show()
    print("RGB 颜色值：")
    for i, c in enumerate(colors):
        print(f"{i+1}: {c.tolist()}")

# 示例
show_palette(r"参考图.jpg")  # 中文路径支持
