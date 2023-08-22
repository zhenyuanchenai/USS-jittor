CUDA='0,1,2,3,4,5,6,7'
N_GPU=8
BATCH=32
DATA=/data/ImageNetS50
IMAGENETS=/data/ImageNetS50

DUMP_PATH=./weights/pass50_sam
DUMP_PATH_FINETUNE=./weights/pass50_sam/pixel_attention
DUMP_PATH_SEG=./weights/pass50_sam/pixel_finetuning
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


mkdir -p ./weights/pass50_sam/pixel_attention
mkdir -p ./weights/pass50_sam/pixel_finetuning

until CUDA_VISIBLE_DEVICES=${CUDA} python res_process/inference_pixel_attention_sam.py -a ${ARCH} \
--pretrained ./weights/pass50/pixel_attention/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ./weights/pass50/pixel_attention/cluster/centroids.npy
do
    echo "error，restarting..."
    sleep 5
done

CUDA_VISIBLE_DEVICES=${CUDA} python res_process/evaluator_sam.py \
--predict_path ${DUMP_PATH_FINETUNE} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation \
--curve \
--min 20 \
--max 80

#until CUDA_VISIBLE_DEVICES=${CUDA} python res_process/inference_pixel_attention_sam.py -a ${ARCH} \
until CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python res_process/inference_pixel_attention_sam.py -a ${ARCH} \
--pretrained ./weights/pass50/pixel_attention/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode train \
--filename train \
--centroid ./weights/pass50/pixel_attention/cluster/centroids.npy \
-t 0.35  #最高的了
do
    echo "error，restarting..."
    sleep 5
done