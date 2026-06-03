#!/bin/bash
#SBATCH --job-name="seg_efficientnet"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=0-4:00
#SBATCH --mail-user=
#SBATCH --mail-type=END,FAIL
#SBATCH --output=./out_seg_efficientnet_%j.out
#SBATCH --error=./err_seg_efficientnet_%j.out

module purge
module load CUDA/12.4
module load Python/Anaconda

source activate dental_seg

export TORCH_HOME=./dental_segmentation/pretrained_weights

# Настройка локального SSD
if [ -n "$SLURM_TMPDIR" ]; then
    LOCAL_SCRATCH="$SLURM_TMPDIR"
else
  cd ..
    LOCAL_SCRATCH="./tmp_slurm_${SLURM_JOB_ID}"
    mkdir -p "$LOCAL_SCRATCH"
fi

# Создаем папку для данных
mkdir -p "$LOCAL_SCRATCH/data"

# Копирование данных
rsync -avh --progress ./dental_segmentation/data/complete_dataset/ "$LOCAL_SCRATCH/data/complete_dataset/"

# Переходим в нужную директорию
cd ./dental_segmentation

# Запуск обучения EfficientNet
echo "=== GPU INFO ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=== START TRAINING (efficientnet)==="
srun python src/train_segmentation.py \
    --data_dir "$LOCAL_SCRATCH/data/complete_dataset" \
    --model_type efficientnet \
    --output_dir "./dental_segmentation/outputs" \
    --batch_size 12 \
    --img_size 512 \
    --epochs 30 \
    --lr 1e-3 \
    --patience 10

echo "=== Конец ==="
