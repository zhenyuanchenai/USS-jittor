CUDA='0,1,2,3,4,5,6,7'
N_GPU=8
BATCH=64
DATA=/data/ImageNetS50
IMAGENETS=/data/ImageNetS50

DUMP_PATH=./weights/pass50_sam_upper50
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning
QUEUE_LENGTH=3840
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet50
NUM_CLASSES=50
EPOCH=200
EPOCH_PIXELATT=20
EPOCH_SEG=20
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python res_process/main_pixel_finetuning_uppernet_50.py \
--arch ${ARCH} \
--data_path ${DATA}/train_new \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 0 \
--num_classes ${NUM_CLASSES} \
--pseudo_path ./weights/pass50_sam/pixel_attention/train \
--pretrained ./weights/pass50_resnet50/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python res_process/inference_sam_point_jittor_seg50.py -a ${ARCH} \
--pretrained ./weights/pass50_sam_upper50/pixel_finetuning/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode testB \