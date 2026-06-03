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
import psutil
import gc
print("=== STEP 5: Utility imports done ===", flush=True)

print("=== SCRIPT READY ===", flush=True)

def print_gpu_memory(): # выводит информацию об использовании GPU памяти
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("  GPU not available")

def print_data_report(loader, name="Dataset"): # Выводит отчет
    print(f"\n{'='*60}")
    print(f" REPORT: {name}")
    print(f"{'='*60}")
    print(f"  Total batches: {len(loader)}")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Total samples: {len(loader.dataset)}")
    print(f"  Num workers: {loader.num_workers}")
    print(f"  Pin memory: {loader.pin_memory}")
    print(f"{'='*60}\n")

def print_epoch_report(epoch, total_epochs, train_loss, val_loss, lr, epoch_time, best_val_loss): # Выводит отчет об эпохе
    print(f"\n{'='*60}")
    print(f" EPOCH {epoch}/{total_epochs} COMPLETED")
    print(f"{'='*60}")
    print(f"  Time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)")
    print(f"  Learning Rate: {lr:.2e}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss: {val_loss:.6f}")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print_gpu_memory()
    print(f"{'='*60}\n")


def compute_segmentation_metrics(pred_masks, true_masks, num_classes): # вычисляет Dice и IoU для сегментации
    import numpy as np

    if torch.is_tensor(pred_masks):
        pred = pred_masks.cpu().numpy()
        true = true_masks.cpu().numpy()
    else:
        pred = pred_masks
        true = true_masks

    dice_scores = []
    iou_scores = []

    for class_id in range(1, num_classes):  # игнорируем фон (класс 0)
        pred_class = (pred == class_id).astype(np.float32)
        true_class = (true == class_id).astype(np.float32)

        intersection = np.sum(pred_class * true_class)
        pred_sum = np.sum(pred_class)
        true_sum = np.sum(true_class)

        # Dice coefficient
        if pred_sum + true_sum == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / (pred_sum + true_sum)

        # IoU (Jaccard Index)
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
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# создание папки для результатов
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.model_type}_{timestamp}"
run_dir = Path(args.output_dir) / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / 'checkpoints').mkdir(exist_ok=True)

print(f"Results will be saved to: {run_dir}")


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

    def create_mask(self, anns, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)  # ← uint8 вместо int64
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

        # ресайз
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # аугментация
        if self.augment and np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # нормализация
        img = img / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()  # ← здесь преобразуем в long для loss

        return img, mask


def split_data(images):  # 80% train, 10% val, 10% test
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from collections import Counter

    for img in images:
        img['patient'] = Path(img['file_name']).stem.split('_')[0]

    patient_to_imgs = defaultdict(list)
    for img in images:
        patient_to_imgs[img['patient']].append(img)

    patients = list(patient_to_imgs.keys())

    # создаем страты
    strata = []
    for patient in patients:
        count = len(patient_to_imgs[patient])
        if count <= 2:
            strata.append('low')
        elif count <= 4:
            strata.append('medium')
        else:
            strata.append('high')

    print(f"Total patients: {len(patients)}")
    print(f"Strata distribution: {Counter(strata)}")

    # 80% train, 20% temp
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(sss.split(patients, strata))

    train_p = [patients[i] for i in train_idx]
    temp_p = [patients[i] for i in temp_idx]

    # temp на val и test
    temp_strata = [strata[patients.index(p)] for p in temp_p]

    print(f"Temp strata distribution: {Counter(temp_strata)}")

    # если в какой-то страте 1 элемент, делаем без стратификации
    if min(Counter(temp_strata).values()) < 2:
        print("Warning: Small stratum detected, splitting without stratification")
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
    else:
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42, stratify=temp_strata)

    train = [img for p in train_p for img in patient_to_imgs[p]]
    val = [img for p in val_p for img in patient_to_imgs[p]]
    test = [img for p in test_p for img in patient_to_imgs[p]]

    print(f"\nResult:")
    print(
        f"  Train: {len(train)} images from {len(train_p)} patients ({len(train_p) / len(patients) * 100:.1f}% of patients)")
    print(
        f"  Val:   {len(val)} images from {len(val_p)} patients ({len(val_p) / len(patients) * 100:.1f}% of patients)")
    print(
        f"  Test:  {len(test)} images from {len(test_p)} patients ({len(test_p) / len(patients) * 100:.1f}% of patients)")

    # проверяем распределение категорий в итоговых выборках
    def get_strata_dist(patients_list):
        counts = {'low': 0, 'medium': 0, 'high': 0}
        for p in patients_list:
            count = len(patient_to_imgs[p])
            if count <= 2:
                counts['low'] += 1
            elif count <= 4:
                counts['medium'] += 1
            else:
                counts['high'] += 1
        return counts

    print("\nDistribution by image count category (patients):")
    print(f"  Train: {get_strata_dist(train_p)}")
    print(f"  Val:   {get_strata_dist(val_p)}")
    print(f"  Test:  {get_strata_dist(test_p)}")

    return train, val, test


def save_split_info(train_imgs, val_imgs, test_imgs, run_dir): # сохраняет информацию о разделении в JSON файл
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
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    print(f"  Test: {len(test_imgs)} images")

    return split_info


def create_model(model_type, num_classes):
    try:
        if model_type == 'resnet18':
            return smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)
        elif model_type == 'efficientnet':
            return smp.Unet('efficientnet-b3', encoder_weights='imagenet', in_channels=3, classes=num_classes)
        elif model_type == 'maxvit':
            # путь к локальным весам
            weights_path = './pretrained_weights/maxvit_small_tf_224_smp_imagenet.pth'

            print(f"Loading MaxViT from local weights: {weights_path}")

            # создаем модель без автоматической загрузки весов
            model = smp.Unet(
                'tu-maxvit_small_tf_224',
                encoder_weights=None,
                in_channels=3,
                classes=num_classes,
                encoder_depth=5,
                decoder_channels=(256, 128, 64, 32, 16)
            )

            # загружаем локальные веса
            if os.path.exists(weights_path):
                print(f"Found weights file, loading...")
                state_dict = torch.load(weights_path, map_location='cpu')

                # адаптируем ключи для encoder
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder.'):
                        new_state_dict[k] = v
                    elif k.startswith('model.'):
                        new_state_dict['encoder.' + k[6:]] = v
                    else:
                        new_state_dict[f'encoder.{k}'] = v

                # загружаем с игнорированием несоответствий
                missing, unexpected = model.encoder.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded encoder weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

                # загружаем также веса для decoder, если есть
                decoder_weights_path = weights_path.replace('_smp_', '_decoder_')
                if os.path.exists(decoder_weights_path):
                    decoder_state = torch.load(decoder_weights_path, map_location='cpu')
                    model.decoder.load_state_dict(decoder_state, strict=False)
                    print("Loaded decoder weights")
            else:
                print(f"WARNING: Weights not found at {weights_path}")
                print("Loading from internet instead...")
                model = smp.Unet(
                    'tu-maxvit_small_tf_224',
                    encoder_weights='imagenet',
                    in_channels=3,
                    classes=num_classes,
                    encoder_depth=5,
                    decoder_channels=(256, 128, 64, 32, 16)
                )

            return model
        else:
            return smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)
    except Exception as e:
        print(f"Error creating model {model_type}: {e}")
        print("Falling back to resnet18")
        return smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, classes=num_classes)


def main():
    # засекаем общее время
    total_start_time = time.time()

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print_gpu_memory()

    print("\nLoading dataset...")
    load_start = time.time()
    images, img_root, cat_map, num_classes = load_data(args.data_dir)
    print(f"  Loaded in {time.time() - load_start:.2f} sec")

    train_imgs, val_imgs, test_imgs = split_data(images)
    print(f"\n DATA SPLIT:")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    print(f"  Test: {len(test_imgs)} images")

    split_info = save_split_info(train_imgs, val_imgs, test_imgs, run_dir)

    print("\nCreating datasets...")
    train_ds = SegmentationDataset(train_imgs, img_root, cat_map, args.img_size, augment=True)
    val_ds = SegmentationDataset(val_imgs, img_root, cat_map, args.img_size, augment=False)
    print(f"  Train dataset: {len(train_ds)} samples")
    print(f"  Val dataset: {len(val_ds)} samples")

    print(f"=== DEBUG: train_ds length = {len(train_ds)}", flush=True)
    print(f"=== DEBUG: val_ds length = {len(val_ds)}", flush=True)

    # gопробуем получить первый элемент для проверки
    try:
        sample_img, sample_mask = train_ds[0]
        print(f"=== DEBUG: Sample image shape = {sample_img.shape}", flush=True)
        print(f"=== DEBUG: Sample mask shape = {sample_mask.shape}", flush=True)
    except Exception as e:
        print(f"=== DEBUG ERROR: {e}", flush=True)

    # cоздаем загрузчики с оптимизациями
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # ускоряет передачу на GPU
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    print(f"num_workers = {args.num_workers}")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # отчет о загрузчиках
    print_data_report(train_loader, "TRAIN LOADER")
    print_data_report(val_loader, "VAL LOADER")

    # МОДЕЛЬ
    print("\n️ Creating model...")
    model_start = time.time()
    model = create_model(args.model_type, num_classes).to(device)
    print(f"  Model created in {time.time() - model_start:.2f} sec")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print_gpu_memory()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(ignore_index=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed Precision (AMP) ENABLED!")
    else:
        scaler = None
        print("Mixed Precision NOT available")

    # ТРЕНИРОВКА
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n" + "=" * 60)
    print(f"STARTING TRAINING FOR {args.epochs} EPOCHS")
    print("=" * 60 + "\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ТРЕНИРОВКА
        model.train()
        train_loss = 0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(imgs), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), masks)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_batches

        # ВАЛИДАЦИЯ
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for imgs, masks in pbar:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss = criterion(model(imgs), masks)
                else:
                    loss = criterion(model(imgs), masks)

                val_loss += loss.item()
                val_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / val_batches

        epoch_time = time.time() - epoch_start

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # отчет об эпохе
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{args.epochs} COMPLETED")
        print(f"{'=' * 60}")
        print(f"Time: {epoch_time:.1f} sec ({epoch_time / 60:.1f} min)")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), run_dir / 'checkpoints' / 'best_model.pth')
            print(f"Saved best model (Val Loss: {best_val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        print_gpu_memory()
        print(f"{'=' * 60}\n")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETED!")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} hours)")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Results saved to: {run_dir}")
    print(f"{'=' * 60}\n")

    # ТЕСТИРОВАНИЕ
    print("\n" + "=" * 60)
    print("TESTING ON TEST SET")
    print("=" * 60)

    # загружаем лучшую модель
    model.load_state_dict(torch.load(run_dir / 'checkpoints' / 'best_model.pth'))
    model.eval()

    # создаем тестовый датасет
    test_ds = SegmentationDataset(test_imgs, img_root, cat_map, args.img_size, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    all_preds = []
    all_masks = []

    print("Running inference on test set...")
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
            else:
                outputs = model(imgs)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # вычисляем метрики
    test_metrics = compute_segmentation_metrics(all_preds, all_masks, num_classes)

    print(f"\nTEST RESULTS:")
    print(f"  Dice Score: {test_metrics['dice']:.4f}")
    print(f"  IoU (Jaccard): {test_metrics['iou']:.4f}")
    print(f"  Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")

    # сохраняем метрики
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # сохраняем полные результаты
    full_results = {
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics,
        'epochs_completed': epoch,
        'early_stopped': patience_counter >= args.patience
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    print(f"   - test_metrics.json")
    print(f"   - results.json")
    print("=" * 60 + "\n")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()