import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil

# конвертация COCO JSON в формат YOLO segmentation
def convert_coco_to_yolo_seg(
        coco_json_path,
        output_dir,
        images_dir=None,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
):

    # загрузка COCO данных
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    print(f"Loaded {len(coco_data['categories'])} categories")
    print(f"Loaded {len(coco_data['images'])} images")
    print(f"Loaded {len(coco_data['annotations'])} annotations")

    # создание маппинга ID категорий
    category_map = {}
    for cat in coco_data['categories']:
        # YOLO использует индексы начиная с 0
        category_map[cat['id']] = cat['name']

    print(f"Categories mapping: {category_map}")

    images_map = {img['id']: img for img in coco_data['images']} # маппинг изображений

    # группировка аннотаций по изображениям
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    output_dir = Path(output_dir)
    train_img_dir = output_dir / 'train' / 'images'
    train_label_dir = output_dir / 'train' / 'labels'
    val_img_dir = output_dir / 'val' / 'images'
    val_label_dir = output_dir / 'val' / 'labels'
    test_img_dir = output_dir / 'test' / 'images'
    test_label_dir = output_dir / 'test' / 'labels'

    # cоздание директорий
    for dir_path in [train_img_dir, train_label_dir, val_img_dir,
                     val_label_dir, test_img_dir, test_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # получение списка изображений с аннотациями
    image_ids = list(annotations_by_image.keys())

    # разделение на train/val/test
    np.random.seed(random_seed)
    np.random.shuffle(image_ids)

    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]

    print(f"\nTotal images with annotations: {n_total}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # функция для нормализации полигона
    def normalize_polygon(polygon, img_width, img_height):
        # нормализация координат полигона в диапазон [0, 1]
        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            # клиппинг
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            normalized.extend([x, y])
        return normalized

    # функция для получения полигона из аннотации
    def get_polygon_from_annotation(ann):
        if 'segmentation' not in ann:
            return None

        seg = ann['segmentation']

        # если это список полигонов
        if isinstance(seg, list):
            # проверяем, не пустой ли список
            if len(seg) == 0:
                return None
            # берем первый полигон
            first_poly = seg[0]
            # проверяем, что это список
            if isinstance(first_poly, list):
                return first_poly
            # если это уже список координат
            elif isinstance(first_poly, (list, tuple)):
                return first_poly

        # если это уже список координат
        elif isinstance(seg, (list, tuple)):
            return seg

        return None

    # функция для сохранения аннотаций в YOLO формат
    def save_yolo_annotations(image_id, annotations, img_info, output_label_dir):
        img_width = img_info['width']
        img_height = img_info['height']

        # имя файла без расширения
        img_filename = Path(img_info['file_name']).stem
        label_path = output_label_dir / f"{img_filename}.txt"

        lines = []
        for ann in annotations:
            # получаем класс (YOLO индексы с 0)
            cat_id = ann['category_id']
            if cat_id not in category_map:
                continue

            class_id = list(category_map.keys()).index(cat_id)

            # получаем полигон
            polygon = get_polygon_from_annotation(ann)

            if polygon is None or len(polygon) < 6:
                print(f"  Warning: Invalid polygon for image {img_info['file_name']}, annotation {ann}")
                continue

            try:
                # нормализуем координаты
                normalized_poly = normalize_polygon(polygon, img_width, img_height)

                # формат YOLO: class_id x1 y1 x2 y2 ...
                poly_str = ' '.join([f"{coord:.6f}" for coord in normalized_poly])
                lines.append(f"{class_id} {poly_str}\n")
            except Exception as e:
                print(f"  Error processing polygon: {e}")
                continue

        # сохраняем только если есть валидные аннотации
        if lines:
            with open(label_path, 'w') as f:
                f.writelines(lines)
        else:
            print(f"  Warning: No valid annotations for {img_info['file_name']}")

    # функция для копирования изображений
    def copy_images(image_ids, output_img_dir, images_dir=None):
        for img_id in tqdm(image_ids, desc=f"Copying to {output_img_dir.parent.name}"):
            img_info = images_map[img_id]

            # определение исходного пути
            if images_dir:
                src_path = Path(images_dir) / img_info['file_name']
            else:
                # предполагаем, что изображения в той же директории, что и JSON
                src_path = Path(coco_json_path).parent / 'images' / img_info['file_name']

            dst_path = output_img_dir / img_info['file_name']

            # копирование только если исходный файл существует
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"  Warning: Image not found: {src_path}")

    # определение директории с изображениями
    if images_dir is None:
        images_dir = Path(coco_json_path).parent / 'images'

    print("\n" + "=" * 60)
    print("Converting dataset to YOLO format...")
    print("=" * 60)

    # обработка TRAIN
    print("\n[Train]")
    copy_images(train_ids, train_img_dir, images_dir)
    for img_id in tqdm(train_ids, desc="Saving annotations"):
        img_info = images_map[img_id]
        annotations = annotations_by_image[img_id]
        save_yolo_annotations(img_id, annotations, img_info, train_label_dir)

    # обработка VAL
    print("\n[Val]")
    copy_images(val_ids, val_img_dir, images_dir)
    for img_id in tqdm(val_ids, desc="Saving annotations"):
        img_info = images_map[img_id]
        annotations = annotations_by_image[img_id]
        save_yolo_annotations(img_id, annotations, img_info, val_label_dir)

    # обработка TEST
    print("\n[Test]")
    copy_images(test_ids, test_img_dir, images_dir)
    for img_id in tqdm(test_ids, desc="Saving annotations"):
        img_info = images_map[img_id]
        annotations = annotations_by_image[img_id]
        save_yolo_annotations(img_id, annotations, img_info, test_label_dir)

    # создание data.yaml файла
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write("# YOLOv11 Dataset Configuration\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(category_map)}\n")
        f.write("names: [")
        f.write(", ".join([f"'{name}'" for name in category_map.values()]))
        f.write("]\n")

    print(f"\nData configuration saved to: {yaml_path}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── data.yaml")
    print(f"    ├── train/")
    print(f"    │   ├── images/ ({len(train_ids)} files)")
    print(f"    │   └── labels/ ({len(train_ids)} files)")
    print(f"    ├── val/")
    print(f"    │   ├── images/ ({len(val_ids)} files)")
    print(f"    │   └── labels/ ({len(val_ids)} files)")
    print(f"    └── test/")
    print(f"        ├── images/ ({len(test_ids)} files)")
    print(f"        └── labels/ ({len(test_ids)} files)")

    return yaml_path


# проверка структуры аннотаций
def inspect_coco_annotations(coco_json_path):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("Inspecting COCO annotations structure")
    print("=" * 60)

    # проверяем несколько аннотаций
    for i, ann in enumerate(data['annotations'][:5]):
        print(f"\nAnnotation {i + 1}:")
        print(f"  image_id: {ann.get('image_id')}")
        print(f"  category_id: {ann.get('category_id')}")
        print(f"  segmentation type: {type(ann.get('segmentation'))}")
        if 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, list) and len(seg) > 0:
                print(f"  first segment length: {len(seg[0]) if seg else 0}")
                print(f"  first segment preview: {seg[0][:10] if seg else 'empty'}")

    return data


if __name__ == "__main__":
    # пути
    COCO_JSON_PATH = "../data/complete_dataset/annotations/annotations.json"
    IMAGES_DIR = "../data/complete_dataset/images"
    OUTPUT_DIR = "../data/yolo_dataset"

    # сначала проверяем структуру аннотаций
    coco_data = inspect_coco_annotations(COCO_JSON_PATH)

    # конвертация
    yaml_path = convert_coco_to_yolo_seg(
        coco_json_path=COCO_JSON_PATH,
        output_dir=OUTPUT_DIR,
        images_dir=IMAGES_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    print(f"\nConversion complete!")
    print(f"   Use: yolo task=segment mode=train data={yaml_path} model=yolo11s-seg.pt")