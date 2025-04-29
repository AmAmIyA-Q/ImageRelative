import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from tqdm import tqdm
import pickle

# ----------- 参数设置 -----------
input_folder = r"加藤光"             # 原图所在的主目录
output_folder = input_folder + r'\ImageRelative'       # 输出目录
thumbnail_size = (200, 200)           # 缩略图大小
method = 'cosine'                     # 'cosine' 或 'euclidean'
reference_image = r'测试.jpg'        # 指定参考图像（设为 None 表示使用第一张）
feature_cache_file = os.path.join(output_folder, 'features.pkl')
# ---------------------------------

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def save_thumbnail(image_path, save_path, size=(200, 200)):
    try:
        if not os.path.exists(save_path):
            image = Image.open(image_path)
            image.thumbnail(size)
            image.save(save_path)
    except Exception as e:
        print(f"无法保存缩略图 {save_path}，原因: {e}")

def compute_similarity(base_feature, features, metric='cosine'):
    if metric == 'cosine':
        return cosine_similarity(base_feature, features)[0]
    elif metric == 'euclidean':
        return -np.linalg.norm(features - base_feature, axis=1)
    else:
        raise ValueError("Unsupported similarity metric")

def generate_html(image_paths, original_paths, output_html):
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write('<html><head><meta charset="utf-8"><title>Color Similarity Album</title></head><body>')
        f.write('<h2>Images sorted by color similarity</h2><div style="display:flex; flex-wrap: wrap;">')
        for thumb_path, full_path in zip(image_paths, original_paths):
            f.write(
                f'<div style="margin:10px; text-align:center;">'
                f'<a href="{full_path}" target="_blank">'
                f'<img src="{thumb_path}" style="width:200px; border:1px solid #ccc;"><br>'
                f'</a>{os.path.basename(full_path)}</div>'
            )
        f.write('</div></body></html>')

def load_or_extract_features(image_files):
    # 读取旧缓存
    if os.path.exists(feature_cache_file):
        with open(feature_cache_file, 'rb') as f:
            cache = pickle.load(f)
        old_files = cache.get('files', [])
        old_features = cache.get('features', [])
        feature_dict = dict(zip(old_files, old_features))
        print(f"已加载缓存特征 {len(old_files)} 条。")
    else:
        feature_dict = {}

    batch_size = 10000
    batch_counter = 0

    for img in tqdm(image_files, desc="提取颜色特征"):
        if img not in feature_dict:
            try:
                feat = extract_color_histogram(img)
                feature_dict[img] = feat
                batch_counter += 1
            except Exception as e:
                print(f"跳过无效图像: {img}，原因: {e}")

        # 每1000张新提取后保存一次
        if batch_counter >= batch_size:
            with open(feature_cache_file, 'wb') as f:
                pickle.dump({'files': list(feature_dict.keys()), 'features': list(feature_dict.values())}, f)
            print(f"已保存中间缓存（总计 {len(feature_dict)} 条）")
            batch_counter = 0

    # 最后保存一次
    with open(feature_cache_file, 'wb') as f:
        pickle.dump({'files': list(feature_dict.keys()), 'features': list(feature_dict.values())}, f)
    print(f"已完成所有特征提取并保存（共 {len(feature_dict)} 条）")

    return list(feature_dict.values())

def main():
    os.makedirs(output_folder, exist_ok=True)
    thumbs_dir = os.path.join(output_folder, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    image_files = []
    for root, dirs, files in tqdm(os.walk(input_folder)):
        # 过滤掉 output_folder
        dirs[:] = [d for d in dirs if os.path.join(root, d) != output_folder]
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
                image_files.append(os.path.join(root, f))


    if len(image_files) == 0:
        print("没有找到任何图像。请确认目录下有图片")
        return

    features = load_or_extract_features(image_files)
    features_np = np.array(features)

    try:
        if reference_image and os.path.exists(reference_image):
            base_feature = extract_color_histogram(reference_image).reshape(1, -1)
            print(f"使用指定图像作为参考: {reference_image}")
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f"参考图像出错：{e}，使用默认第一张图片作为参考。")
        base_feature = features_np[0].reshape(1, -1)

    similarities = compute_similarity(base_feature, features_np, metric=method)
    sorted_indices = np.argsort(-similarities)

    print("生成缩略图中...")
    sorted_thumbnails = []
    for idx in tqdm(sorted_indices):
        img_path = image_files[idx]
        thumb_name = os.path.basename(img_path)
        thumb_full = os.path.join(thumbs_dir, thumb_name)
        save_thumbnail(img_path, thumb_full, thumbnail_size)
        thumb_rel = os.path.relpath(thumb_full, output_folder)
        sorted_thumbnails.append(thumb_rel)

    print("生成 HTML 中...")
    html_path = os.path.join(output_folder, '点击查看排序结果.html')
    sorted_originals = [os.path.relpath(image_files[idx], output_folder) for idx in sorted_indices]
    generate_html(sorted_thumbnails, sorted_originals, html_path)

    print("完成！HTML 输出至：", html_path)

if __name__ == "__main__":
    main()