#train_segmentation.py с albumentations аугментациями

import os

print("=== STEP 1: Starting imports ===", flush=True)
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings

print("=== STEP 2: Basic imports done ===", flush=True)

import numpy as np
import cv2
import torch

print("=== STEP 3: PyTorch imported ===", flush=True)

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
from losses import CombinedLoss

print("=== STEP 4: All imports done ===", flush=True)

import time
import gc

print("=== STEP 5: Utility imports done ===", flush=True)

# добавляем albumentations
try:
    import albumentations as A

    print("=== Albumentations imported successfully ===", flush=True)
    ALBUMS_AVAILABLE = True
except ImportError:
    print("=== Albumentations not available, using basic augmentation ===", flush=True)
    ALBUMS_AVAILABLE = False

print("=== SCRIPT READY ===", flush=True)

# получение аугментаций для тренировки
def get_train_augmentations(img_size=512):
    if not ALBUMS_AVAILABLE:
        return None

    return A.Compose([
        # пространственные аугментации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),

        # аугментации цвета
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
        #A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),

        # шум и артефакты
        #A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        #A.GaussianBlur(blur_limit=(3, 5), p=0.1),

        # адаптивный ресайз
        A.Resize(img_size, img_size),
    ])

# аугментации для валидации (только ресайз)
def get_val_augmentations(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
    ]) if ALBUMS_AVAILABLE else None


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f"  GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("  GPU not available")


def print_data_report(loader, name="Dataset"):
    print(f"\n{'=' * 60}")
    print(f" REPORT: {name}")
    print(f"{'=' * 60}")
    print(f"  Total batches: {len(loader)}")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Total samples: {len(loader.dataset)}")
    print(f"  Num workers: {loader.num_workers}")
    print(f"  Pin memory: {loader.pin_memory}")
    print(f"{'=' * 60}\n")


# метрики
def compute_segmentation_metrics(pred_masks, true_masks, num_classes):
    import numpy as np

    if torch.is_tensor(pred_masks):
        pred = pred_masks.cpu().numpy()
        true = true_masks.cpu().numpy()
    else:
        pred = pred_masks
        true = true_masks

    dice_scores = []
    iou_scores = []

    for class_id in range(1, num_classes):
        pred_class = (pred == class_id).astype(np.float32)
        true_class = (true == class_id).astype(np.float32)

        intersection = np.sum(pred_class * true_class)
        pred_sum = np.sum(pred_class)
        true_sum = np.sum(true_class)

        if pred_sum + true_sum == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / (pred_sum + true_sum)

        union = pred_sum + true_sum - intersection
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        dice_scores.append(dice)
        iou_scores.append(iou)

    return {
        'dice': float(np.mean(dice_scores)),
        'iou': float(np.mean(iou_scores)),
        'pixel_accuracy': float(np.mean(pred == true))
    }


# парсер аргументов
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/complete_dataset')
parser.add_argument('--model_type', type=str, default='resnet18')
parser.add_argument('--output_dir', type=str, default='./outputs')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--use_albumentations', action='store_true', default=True,
                    help='Use albumentations for data augmentation')
args = parser.parse_args()

# устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# создание папки для результатов
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.model_type}_{timestamp}_aug"
run_dir = Path(args.output_dir) / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / 'checkpoints').mkdir(exist_ok=True)

print(f"Results will be saved to: {run_dir}")


# загрузка данных
def load_data(data_dir):
    json_path = Path(data_dir) / 'annotations' / 'annotations.json'
    img_root = Path(data_dir) / 'images'

    with open(json_path) as f:
        data = json.load(f)

    cat_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
    num_classes = len(data['categories'])

    img_to_ann = defaultdict(list)
    for ann in data['annotations']:
        img_to_ann[ann['image_id']].append(ann)

    images = []
    for img in data['images']:
        if img['id'] in img_to_ann:
            images.append({
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'anns': img_to_ann[img['id']]
            })

    print(f"Loaded {len(images)} images, {num_classes} classes")
    return images, img_root, cat_map, num_classes + 1


class SegmentationDataset(Dataset):
    def __init__(self, images, img_root, cat_map, img_size, augment=False):
        self.images = images
        self.img_root = img_root
        self.cat_map = cat_map
        self.img_size = img_size
        self.augment = augment

        # создаем аугментации
        if augment and args.use_albumentations and ALBUMS_AVAILABLE:
            self.transform = get_train_augmentations(img_size)
            print("Using albumentations for training")
        elif augment:
            self.transform = None
            print("Using basic augmentation (flip only)")
        else:
            self.transform = get_val_augmentations(img_size) if ALBUMS_AVAILABLE else None

    def create_mask(self, anns, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] in self.cat_map:
                cls = self.cat_map[ann['category_id']] + 1
                if isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], cls)
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.img_root / info['file_name']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.create_mask(info['anns'], info['height'], info['width'])

        # применяем аугментации
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            # поворот
            if self.augment and np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            # ресайз
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # нормализация
        img = img / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask


def split_data(images):
    from sklearn.model_selection import train_test_split

    for img in images:
        img['patient'] = Path(img['file_name']).stem.split('_')[0]

    patient_to_imgs = defaultdict(list)
    for img in images:
        patient_to_imgs[img['patient']].append(img)

    patients = list(patient_to_imgs.keys())

    # страты по количеству изображений
    strata = []
    for patient in patients:
        count = len(patient_to_imgs[patient])
        if count <= 2:
            strata.append('low')
        elif count <= 4:
            strata.append('medium')
        else:
            strata.append('high')

    # 80% train, 20% temp
    train_p, temp_p = train_test_split(
        patients,
        test_size=0.2,
        random_state=42,
        stratify=strata
    )

    # temp на val и test
    temp_strata = [strata[patients.index(p)] for p in temp_p]

    # проверяем, есть ли в каждой страте минимум 2 элемента
    from collections import Counter
    strata_counts = Counter(temp_strata)

    if min(strata_counts.values()) < 2:
        print("Warning: Small stratum detected, splitting without stratification")
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
    else:
        val_p, test_p = train_test_split(
            temp_p,
            test_size=0.5,
            random_state=42,
            stratify=temp_strata
        )

    train = [img for p in train_p for img in patient_to_imgs[p]]
    val = [img for p in val_p for img in patient_to_imgs[p]]
    test = [img for p in test_p for img in patient_to_imgs[p]]

    print(f"Total patients: {len(patients)}")
    print(f"Train: {len(train)} images from {len(train_p)} patients")
    print(f"Val: {len(val)} images from {len(val_p)} patients")
    print(f"Test: {len(test)} images from {len(test_p)} patients")

    return train, val, test


def save_split_info(train_imgs, val_imgs, test_imgs, run_dir):
    split_info = {
        'train': [img['file_name'] for img in train_imgs],
        'val': [img['file_name'] for img in val_imgs],
        'test': [img['file_name'] for img in test_imgs],
        'stats': {
            'train_count': len(train_imgs),
            'val_count': len(val_imgs),
            'test_count': len(test_imgs),
            'total_count': len(train_imgs) + len(val_imgs) + len(test_imgs)
        }
    }

    split_path = run_dir / 'split_info.json'
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Split info saved to: {split_path}")
    return split_info


def create_model(model_type, num_classes):
    try:
        if model_type == 'resnet18':
            return smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)
        elif model_type == 'efficientnet':
            return smp.Unet('efficientnet-b3', encoder_weights='imagenet', in_channels=3, classes=num_classes)
        elif model_type == 'maxvit':
            return smp.Unet('efficientnet-b3', encoder_weights='imagenet', in_channels=3, classes=num_classes)
        else:
            return smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)
    except Exception as e:
        print(f"Error creating model {model_type}: {e}")
        return smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)


def main():
    print("\n=== CUDA DEBUG INFO ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print("=======================\n")

    total_start_time = time.time()

    print("\n" + "=" * 60)
    print("STARTING TRAINING WITH ALBUMENTATIONS")
    print("=" * 60)

    print("\nLoading dataset...")
    images, img_root, cat_map, num_classes = load_data(args.data_dir)

    train_imgs, val_imgs, test_imgs = split_data(images)
    save_split_info(train_imgs, val_imgs, test_imgs, run_dir)

    print("\nCreating datasets with augmentation...")
    train_ds = SegmentationDataset(train_imgs, img_root, cat_map, args.img_size, augment=True)
    val_ds = SegmentationDataset(val_imgs, img_root, cat_map, args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # МОДЕЛЬ
    print("\n Creating model...")
    model = create_model(args.model_type, num_classes).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(ignore_index=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed Precision (AMP) ENABLED!")

    # ТРЕНИРОВКА
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ТРЕНИРОВКА
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), masks)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # ВАЛИДАЦИЯ
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for imgs, masks in pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = criterion(model(imgs), masks)
                else:
                    loss = criterion(model(imgs), masks)
                val_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"\nEPOCH {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), run_dir / 'checkpoints' / 'best_model.pth')
            print(f"  Saved best model (Val Loss: {best_val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTRAINING COMPLETED! Best Val Loss: {best_val_loss:.6f}")

    # ТЕСТИРОВАНЕ
    print("\n" + "=" * 60)
    print("TESTING ON TEST SET")
    print("=" * 60)

    model.load_state_dict(torch.load(run_dir / 'checkpoints' / 'best_model.pth'))
    model.eval()

    test_ds = SegmentationDataset(test_imgs, img_root, cat_map, args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    all_preds, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    test_metrics = compute_segmentation_metrics(all_preds, all_masks, num_classes)

    print(f"\nTEST RESULTS:")
    print(f"  Dice Score: {test_metrics['dice']:.4f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")

    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    full_results = {
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics,
        'used_albumentations': ALBUMS_AVAILABLE and args.use_albumentations
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()