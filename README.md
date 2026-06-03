# Dental Image Segmentation
<div align="center">
**Семантическая и инстанс-сегментация стоматологических изображений с использованием U-Net и YOLOv11**
</div>

## Описание

Проект посвящен сегментации стоматологических изображений (интраоральных фотографий) с выделением 12 классов анатомических структур:
- Teeth (зубы)
- Gums (десны)
- Tongue (язык)
- Cheeks (щеки)
- Hard palate (твердое небо)
- Lips (губы)
- Face (лицо)
- Braces (брекеты)
- Implants (импланты)
- Medical equipment (медицинское оборудование)
- Floor of mouth (дно полости рта)

## Архитектуры и результаты

### Семантическая сегментация (U-Net)
| Энкодер | Dice | IoU |
|---------|------|-----|
| ResNet18 | 0.7365 | 0.6272 |
| EfficientNet-B3 | **0.7894** | **0.6791** |
| MaxViT | 0.7253 | 0.6268 |

### Инстанс-сегментация (YOLOv11)
| Модель | mAP50 |
|--------|-------|
| YOLOv11s-seg | **0.821** (Teeth) |

### Сравнение аугментаций (EfficientNet-B3)
| Тип аугментаций | Dice | IoU | Precision | Recall |
|-----------------|------|-----|-----------|--------|
| Базовые (flip) | 0.7588 | 0.6494 | 0.7634 | 0.7542 |
| Пространственные | **0.7894** | **0.6791** | **0.7932** | **0.7856** |
| Полные (с цветовыми) | 0.7011 | 0.5942 | 0.7123 | 0.6901 |

### Per-class результаты YOLOv11
| Класс | mAP50 |
|-------|-------|
| Floor of mouth | 0.854 |
| Hard palate | 0.832 |
| Teeth | **0.821** |
| Face | 0.798 |
| Medical equipment | 0.714 |
| Lips | 0.602 |

## Статистика
- Изображений: 182
- Аннотаций: 2390
- Классов: 12

## Структура
```
dental_segmentation/
├── src/
│ ├── train_segmentation.py # обучение семантической сегментации
│ ├── train_segmentation_albumentations.py # обучение семантической сегментации с albumentations
│ ├── losses.py # функции потерь
│ └── convert_coco_to_yolo_seg.py # конвертация COCO → YOLO
├── scripts/ # SLURM скрипты
│ ├── run_resnet18.slurm # запуск ResNet18
│ ├── run_efficientnet_aug.slurm # запуск EfficientNet с аугментациями
│ ├── run_maxvit.slurm # запуск MaxViT
│ └── run_yolo.sh # запуск YOLOv11
├── data/
│   └── complete_dataset/ # все изображения
│       ├── images/
│       ├── annotations/
│       │   └── annotations.json # COCO формат
│       └── classes.json # список классов
├── requirements.txt
└── README.md
```

## Установка
1. **Создание виртуального окружения**

Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

2. **Установка зависимостей**

```bash
pip install -r requirements.txt
```

3. **Подготовка данных**

Конвертация COCO аннотаций в YOLO формат:
```bash
python src/convert_coco_to_yolo_seg.py \
    --coco_json data/annotations/annotations.json \
    --output_dir data/yolo_dataset
```

4. **Скачивание предобученных весов**

В папку dental_segmentation/pretrained_weights.
resnet18: https://download.pytorch.org/models/resnet18-f37072fd.pth
efficientnet_b3: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3-933fb5bb.pth
maxvit_small_tf_224: https://huggingface.co/timm/maxvit_small_tf_224/resolve/main/model.safetensors

6. **Обучение моделей**

#### U-Net с EfficientNet-B3 (аналогично для ResNet18 и MaxVit):
```bash
python src/train_segmentation_v2.py \
    --data_dir data/complete_dataset \
    --model_type efficientnet \
    --batch_size 8 \
    --img_size 512 \
    --epochs 50 \
    --lr 1e-3 \
    --patience 15 \
    --use_albumentations
```

Запуск на GPU-кластере:
```bash
sbatch scripts/run_efficientnet_aug.slurm
```

#### YOLOv11 инстанс-сегментация
```bash
yolo task=segment mode=train \
    data=data/yolo_dataset/data.yaml \
    model=yolo11s-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0
```

