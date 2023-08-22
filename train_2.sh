CUDA='0,1,2,3,4,5,6,7'
N_GPU=8
BATCH=32
DATA=/data/ImageNetS50
IMAGENETS=/data/ImageNetS50

DUMP_PATH=./weights/pass50
DUMP_PATH_FINETUNE=./weights/pass50/pixel_attention
DUMP_PATH_SEG=./weights/pass50/pixel_finetuning
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet18
NUM_CLASSES=50
EPOCH=200
EPOCH_PIXELATT=20
EPOCH_SEG=20
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0


mkdir -p ./weights/pass50/pixel_attention
mkdir -p ./weights/pass50/pixel_finetuning

#不需要阈值处理的操作
CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python res_process/main_pretrain.py \
--arch resnet18 \
--data_path /data/ImageNetS50/train \
--dump_path ./weights/pass50 \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp 512 \
--nmb_prototypes 500 \
--queue_length 2048 \
--epoch_queue_starts 15 \
--epochs 200 \
--batch_size 64 \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters 1001 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 0 \
--seed 31 \
--shallow 3 \
--weights 1 1

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python res_process/main_pixel_attention.py \
--arch ${ARCH} \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH_PIXELATT} \
--epoch_queue_starts 0 \
--epochs ${EPOCH_PIXELATT} \
--batch_size ${BATCH} \
--base_lr 6.0 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES_PIXELATT} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 4 \
--seed 31 \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar \
--threshold 0.34

CUDA_VISIBLE_DEVICES=${CUDA} python res_process/cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--threshold 0.34


