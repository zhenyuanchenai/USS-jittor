import sys
sys.path.insert(0, '/data/USS-jittor/')
import os
import argparse
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
import numpy as np
from tqdm import tqdm
import jittor.transform as transforms
from PIL import Image
import src.resnet_uppernet50 as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import bool_flag

from src.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--filename", type=str, required=False)
    parser.add_argument("--dump_path", type=str, default=None, help="The path to save results.")
    parser.add_argument("--data_path", type=str, default=None, help="The path to ImagenetS dataset.")
    parser.add_argument("--pretrained", type=str, default=None, help="The model checkpoint file.")
    parser.add_argument("-a", "--arch", metavar="ARCH", help="The model architecture.")
    parser.add_argument("-c", "--num-classes", default=50, type=int, help="The number of classes.")
    parser.add_argument("-t", "--threshold", default=0, type=float, help="The threshold to filter the 'others' categroies.")
    parser.add_argument("--test", action='store_true', help="whether to save the logit. Enabled when finding the best threshold.")
    parser.add_argument("--centroid", type=str, default=None, help="The centroids of clustering.")
    parser.add_argument("--checkpoint_key", type=str, default='state_dict', help="key of model in checkpoint")

    args = parser.parse_args()

    return args


def main_worker(args):

    centroids = np.load(args.centroid)  #加载中心
    centroids = jt.array(centroids)    
    centroids = jt.normalize(centroids, dim=1, p=2)  #中心特征归一化

    sam = sam_model_registry["vit_b"](checkpoint="/data/SAM/sam_vit_b_01ec64.pth.tar")
    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,   #就是64
        # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing       
    )

    # build model
    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')       #建立模型，不做任何投影
    else:
        raise NotImplementedError()

    checkpoint = jt.load(args.pretrained)[args.checkpoint_key]
    for k in list(checkpoint.keys()):
        if k.startswith('module.'):
            checkpoint[k[len('module.'):]] = checkpoint[k]
            del checkpoint[k]
            k = k[len('module.'):]
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    model.eval()

    # build dataset
    if args.mode == "validation":
        data_path = os.path.join(args.data_path, args.mode)
    else:
        data_path = os.path.join(args.data_path, args.filename)
    normalize = transforms.ImageNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.mode == "validation":
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
        dataloader = dataset.set_attrs(batch_size=1, num_workers=4, drop_last=False, shuffle=False)
    else:
        dataset = InferImageFolder(root=data_path,
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.ToTensor(),
                            normalize,
                        ]),                               
                            transform_raw = transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                        ]),
                        num_gpus=jt.world_size)
        dataloader = dataset.set_attrs(batch_size=jt.world_size, num_workers=0, drop_last=False, shuffle=False)


    dump_path = os.path.join(args.dump_path, args.mode)

    if not jt.in_mpi or (jt.in_mpi and jt.rank == 0):
        for cate in os.listdir(data_path):
            if not os.path.exists(os.path.join(dump_path, cate)):
                os.makedirs(os.path.join(dump_path, cate))

    for images_raw, images, path, height, width in tqdm(dataloader):
        path = path[0]
        cate = path.split("/")[-2]
        name = path.split("/")[-1].split(".")[0]

        with jt.no_grad():
            # if not os.path.exists(os.path.join(dump_path, cate, name + ".png")):
            #     print("FUCK",os.path.join(dump_path, cate, name + ".png"))
            h = height.item()
            w = width.item()

            out, mask = model(images, mode='inference_pixel_attention')  
            
            mask = nn.upsample(mask, (h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)      #插值

            out = jt.normalize(out, dim=1, p=2)       
            B, C, H, W = out.shape

            out_inter =  nn.interpolate(out, (h, w), mode="bilinear", align_corners=False)
            img_raw = cv2.imread(path)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            annotations = mask_branch_model.generate(img_raw)
            annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)  
            
            # out = out.view(B, C, -1).permute(0, 2, 1).view(-1, C)   #只留下通道，其他都变成一个单独的元素
            # cosine = jt.matmul(out, centroids.t())                 #每一个元素跟中心算余弦距离
            # cosine = cosine.view(1, H, W, args.num_classes).permute(0, 3, 1, 2)
            # prediction = jt.argmax(cosine, dim=1, keepdims=True)[0] + 1           #余弦距离最大的那个就是预测结果（前景的预测结果）one-hot
            # prediction = nn.interpolate(prediction.float(), (h, w), mode="nearest").squeeze(0).squeeze(0)     #前景预测结果插值到图像大小。
            
            prediction = jt.zeros((h,w))
            logits_all = mask

            # try:
            for ann in annotations:
                valid_mask = jt.array(ann['segmentation'])
                if valid_mask.shape != logits_all.shape and valid_mask.transpose(-1,-2).shape == logits_all.shape:
                    valid_mask = valid_mask.transpose(-1,-2)
                    print(images.shape,img_raw.shape[:2])
                
                out_mask = out_inter[:,:,valid_mask].mean(dim=-1)
                mask_mean = logits_all[valid_mask].mean(dim=-1)
                

                expanded_pred_mask = jt.ones(logits_all.shape) * mask_mean        # 使用 valid_mask 更新 prediction 

                logits_all = logits_all * (1 - valid_mask) + expanded_pred_mask * valid_mask

                #logits_all[valid_mask] = mask_mean[0]
                
                cosine = jt.matmul(out_mask, centroids.t())                 #每一个元素跟中心算余弦距离
                pred = jt.argmax(cosine, dim=1, keepdims=True)[0] + 1           #余弦距离最大的那个就是预测结果（前景的预测结果）one-hot
                
                expanded_pred = jt.ones(prediction.shape) * pred[0]        # 使用 valid_mask 更新 prediction 
                prediction = prediction * (1 - valid_mask) + expanded_pred * valid_mask
                #prediction[valid_mask] = pred[0]

                del valid_mask, out_mask, mask_mean, cosine, pred, expanded_pred, expanded_pred_mask

                    #prediction[valid_mask] = pred[0]            
                    #del valid_mask,out_mask,cosine,pred,mask_mean,expanded_pred
            # except:
            #     for ann in annotations:
            #         valid_mask = jt.array(ann['segmentation']).data
            #         if valid_mask.shape != logits_all.shape and valid_mask.transpose(-1,-2).shape == logits_all.shape:
            #             valid_mask = valid_mask.transpose(-1,-2)
            #             print(images.shape,img_raw.shape[:2])
                    
            #         out_inter = out_inter.data
            #         out_mask = out_inter[:,:,valid_mask].mean()
            #         logits_all = logits_all.data
            #         mask_mean = logits_all[valid_mask].mean()
                    

            #         expanded_pred_mask = jt.ones(logits_all.shape) * mask_mean        # 使用 valid_mask 更新 prediction 

            #         logits_all = logits_all * (1 - valid_mask) + expanded_pred_mask.data * valid_mask

            #         #logits_all[valid_mask] = mask_mean[0]
                    
            #         cosine = jt.matmul(jt.array(out_mask), centroids.t())                 #每一个元素跟中心算余弦距离
            #         pred = jt.argmax(cosine, dim=1, keepdims=True)[0] + 1           #余弦距离最大的那个就是预测结果（前景的预测结果）one-hot
                    
            #         expanded_pred = jt.ones(prediction.shape) * pred[0]        # 使用 valid_mask 更新 prediction 
            #         prediction = prediction * (1 - valid_mask) + expanded_pred.data * valid_mask
            #         #prediction[valid_mask] = pred[0]

            #         del valid_mask, out_mask, mask_mean, cosine, pred, expanded_pred, expanded_pred_mask

            #         #prediction[valid_mask] = pred[0]            
            #         #del valid_mask,out_mask,cosine,pred,mask_mean,expanded_pred
            #         logits_all = jt.array(logits_all)
            #         prediction = jt.array(prediction)

            #del annotations
            #logit = copy.deepcopy(mask)
            prediction[logits_all < args.threshold] = 0

            res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))           #
            res[:, :, 0] = prediction % 256                                         #背景
            res[:, :, 1] = prediction // 256                                        #感觉还是说做了一个颜色上去。

            res = res.data
            #logit = logit.data
            
            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + ".png"))                 #
            if args.test:
                logits_all = logits_all.data
                #np.save(os.path.join(dump_path, cate, name + ".npy"), logit)
                np.save(os.path.join(dump_path, cate, name + "lopgits_all.npy"), logits_all)
            
            jt.clean_graph()
            jt.sync_all()
            #jt.display_memory_info()
            jt.gc()


if __name__ == "__main__":
    args = parse_args()
    main_worker(args=args)