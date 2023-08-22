import sys
sys.path.insert(0, '/data/USS-jittor/')
import argparse
import json
import os
import shutil

import numpy as np
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
from PIL import Image
import jittor.transform as transforms
from tqdm import tqdm

import src.resnet_uppernet50 as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import hungarian

from src.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pycocotools.mask as maskUtils
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--dump_path',
                        type=str,
                        default=None,
                        help='The path to save results.')
    parser.add_argument('--match_file',
                        type=str,
                        default=None,
                        help='The matching file for test set.')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='The path to ImagenetS dataset.')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='The model checkpoint file.')
    parser.add_argument('--pretrained_sam',
                        type=str,
                        default="/data/SAM/sam_vit_b_01ec64.pth.tar",
                        help='The sam checkpoint file.')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        help='The model architecture.')
    parser.add_argument('-c',
                        '--num-classes',
                        default=50,
                        type=int,
                        help='The number of classes.')
    parser.add_argument('--max_res', default=1000, type=int, help="Maximum resolution for evaluation. 0 for disable.")
    parser.add_argument('--method',
                        default='example submission',
                        help='Method name in method description file(.txt).')
    parser.add_argument('--train_data',
                        default='null',
                        help='Training data in method description file(.txt).')
    parser.add_argument(
        '--train_scheme',
        default='null',
        help='Training scheme in method description file(.txt), \
            e.g., SSL, Sup, SSL+Sup.')
    parser.add_argument(
        '--link',
        default='null',
        help='Paper/project link in method description file(.txt).')
    parser.add_argument(
        '--description',
        default='null',
        help='Method description in method description file(.txt).')
    args = parser.parse_args()

    return args

def split_and_unique(mask, block_size):
    # 获取 mask 的 shape
    h, w = mask.shape
    
    # 初始化一个空列表来存储 unique 值
    unique_values = []

    # 遍历 mask 中的每个 block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # 提取当前 block
            block = mask[i:i+block_size, j:j+block_size]

            # 找到 block 中的 unique values 并添加到 list 中
            unique_values.append(jt.misc.unique(block))

    return unique_values

def main_worker(args):
    # build model
    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](
            hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=args.num_classes)
    else:
        raise NotImplementedError()

    checkpoint = jt.load(args.pretrained)["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    model.eval()

    sam = sam_model_registry["vit_b"](checkpoint=args.pretrained_sam)
    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,   #就是64
        # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        output_mode='coco_rle',        
    )
    predictor = SamPredictor(sam)
    # build dataset
    assert args.mode in ['train','validation', 'testA', 'testB']
    data_path = os.path.join(args.data_path, args.mode)
    validation_segmentation = os.path.join(args.data_path,
                                           'validation-segmentation')
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = InferImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.ToTensor(),
                                   normalize,
                               ]),
                               transform_raw = transforms.Compose([
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    dataloader = dataset.set_attrs(
        batch_size=1, 
        num_workers=16)

    dump_path = os.path.join(args.dump_path, args.mode)

    targets = []
    predictions = []
    for images_raw, images, path, height, width in tqdm(dataloader):
        path = path[0]
        cate = path.split('/')[-2]
        name = path.split('/')[-1].split('.')[0]
        if not os.path.exists(os.path.join(dump_path, cate)):
            os.makedirs(os.path.join(dump_path, cate))
        #if not os.path.exists(os.path.join(dump_path, cate, name + ".png")):
        with jt.no_grad():
            H = height.item()
            W = width.item()    
            
            predictor.set_image(images_raw[0,...].data)

            output = model(images)
            output_global = jt.mean(output[:,1:,:,:],dims=(2,3))
            prediction_global = jt.argmax(output_global, dim=1, keepdims=True)[0]  
            sam_mask_input = nn.interpolate(output, (256, 256), mode="bilinear", align_corners=False)

            output = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
            prediction = jt.argmax(output, dim=1, keepdims=True)[0]        
            
            prediction = prediction.squeeze(0).squeeze(0)
            cls_pred_index_pure =  np.unique(prediction.data)
            #semantc_mask = copy.deepcopy(prediction)
            semantc_mask = prediction
            try:
                annotations = mask_branch_model.generate(np.array(images_raw)[0,...])
                annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
                for ann in annotations:
                    valid_mask = jt.array(maskUtils.decode(ann['segmentation'])).bool()
                    # get the class ids of the valid pixels
                    propose_classes_ids = prediction[valid_mask]
                    num_class_proposals = len(np.unique(propose_classes_ids.data))    
                    if num_class_proposals == 1:
                        semantc_mask[valid_mask] = propose_classes_ids[0]
                        continue

                    top_1_propose_class_ids = jt.misc.topk(jt.array(np.bincount(np.array(propose_classes_ids.flatten()))),1)[1]
                    semantc_mask[valid_mask] = top_1_propose_class_ids
                    
                    del valid_mask, propose_classes_ids, num_class_proposals, top_1_propose_class_ids
            except:
                print("no sam")

            cls_pred_index = np.unique(semantc_mask.data)   
            
            if not all(x == 0 for x in cls_pred_index):
                res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
                res[:, :, 0] = semantc_mask % 256
                res[:, :, 1] = semantc_mask // 256
                res = res.data
            elif not all(x == 0 for x in cls_pred_index_pure):
                res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
                res[:, :, 0] = prediction % 256
                res[:, :, 1] = prediction // 256
                res = res.data   
            else:
                max_indices = jt.argmax(sam_mask_input, dim=1)[0]
                # 初始化一个全False的布尔矩阵（大小为(B, H, W)）
                max_mask = jt.zeros_like(max_indices)
                # 使用固定的index值而不是遍历C维度中的索引
                index = int(prediction_global.data)+1
                # 创建临时布尔矩阵，表示当前索引是否为最大值
                temp_mask = max_indices == index
                # 对应位置为True，则更新最终矩阵
                max_mask = max_mask | (temp_mask & (sam_mask_input[:, index, :, :] == sam_mask_input.max(dim=1).unsqueeze(1)))

                selected_values = output[0,(int(prediction_global.data)+1),:,:].data
                sorted_indices = np.argsort(selected_values, axis=None)
                input_point = []
                input_label = []
                # Select k smallest
                for i in range(5):
                    point_min = list(np.unravel_index(sorted_indices[i], selected_values.shape))
                    input_point.append(point_min)
                    input_label.append(0)

                # Select k largest
                for i in range(-5, 0):
                    point_max = list(np.unravel_index(sorted_indices[i], selected_values.shape))
                    input_point.append(point_max)
                    input_label.append(1)
                    
                masks,_,_ = predictor.predict(
                    point_coords=np.array(input_point),
                    point_labels=np.array(input_label),
                    mask_input=max_mask[0,:,:,:].data,
                    multimask_output=False,
                )
                
                label_mask = jt.array((int(prediction_global.data)+1)).view(1,1,1)
                result = jt.array(masks).long() * label_mask
                final_mask = result.squeeze(0)    
                
                res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
                res[:, :, 0] = final_mask % 256
                res[:, :, 1] = final_mask // 256
                res = res.data              

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + '.png'))

            if args.mode == 'validation':
                target = Image.open(os.path.join(validation_segmentation, cate, name + '.png'))
                target = np.array(target).astype(np.int32)
                target = target[:, :, 1] * 256 + target[:, :, 0]

                # Prepare for matching (target)
                target_unique = np.unique(target.reshape(-1))
                target_unique = target_unique - 1
                target_unique = target_unique.tolist()
                if -1 in target_unique:
                    target_unique.remove(-1)
                targets.append(target_unique)

                # Prepare for matching (prediction)
                prediction_unique = np.unique(semantc_mask.data.reshape(-1))
                prediction_unique = prediction_unique - 1
                prediction_unique = prediction_unique.tolist()
                if -1 in prediction_unique:
                    prediction_unique.remove(-1)
                predictions.append(prediction_unique)
            
            jt.clean_graph()
            jt.sync_all()
            jt.gc()
                
    if args.mode == 'validation':    
        _, match = hungarian(targets, predictions, num_classes=args.num_classes)
        match = {k + 1: v + 1 for k, v in match.items()}
        match[0] = 0

        with open(os.path.join(dump_path, 'match.json'), 'w') as f:
            f.write(json.dumps(match))

    elif args.mode == 'testA' or args.mode == 'testB':
        # assert os.path.exists(args.match_file)
        # shutil.copyfile(args.match_file, os.path.join(dump_path, 'match.json'))

        # method = 'Method name: {}\n'.format(args.method) + \
        #     'Training data: {}\nTraining scheme: {}\n'.format(
        #         args.train_data, args.train_scheme) + \
        #     'Networks: {}\nPaper/Project link: {}\n'.format(
        #         args.arch, args.link) + \
        #     'Method description: {}'.format(args.description)
        # with open(os.path.join(dump_path, 'method.txt'), 'w') as f:
        #     f.write(method)

        # zip for submission
        shutil.make_archive(os.path.join(args.dump_path, "result"), 'zip', root_dir=dump_path)


if __name__ == '__main__':
    args = parse_args()
    main_worker(args=args)

