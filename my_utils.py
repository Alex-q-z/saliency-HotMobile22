import warnings
warnings.filterwarnings('ignore')
import IPython.display
import argparse
import logging
import os
import copy
import subprocess
from pathlib import Path
from munch import Munch
import glob
import pprint
import pickle
import time
from datetime import datetime
import coloredlogs
import matplotlib.pyplot as plt
from itertools import product
import enlighten
import random
import gc
import cv2
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import io
from dnn.CARN.interface import CARN
from dnn.dnn_factory import DNN_Factory
from utils.mask_utils import merge_black_bkgd_images
from utils.results_utils import write_results
from utils.timer import Timer
from utils.video_utils import read_videos
from utils.results_utils import read_ground_truth, read_results
from utils.bbox_utils import *
from utils.visualize_utils import visualize_heat_qz, visualize_heat
from utils.mask_utils import tile_mask

class Video_Dataset(Dataset):
    def __init__(self, video_frames_path):
        self.path = video_frames_path
        self.len = len(glob.glob(self.path + "*.png"))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = PIL.Image.open(self.path + "%010d.png" % idx).convert("RGB")

        w, h = image.size
        if h > w:
            return None
        transform_hq = T.Compose(
            [
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )

        return {
            "image_rgb": np.array(image).astype(np.int32),
            "image": transform_hq(image),
            "fid": idx,
            "video_name": "trafficcam",
        }

def compute_loss(my_result, gt_result, car_only = False):

    IoU_threshold = 0.5
    threshold = 0.5
    if car_only:
        valid_index_gt = (gt_result[0]['scores'] > threshold) & (gt_result[0]['labels'] == 3)
    else:
        valid_index_gt = (gt_result[0]['scores'] > threshold)
    
    my_boxes = my_result[0]['boxes']
    my_scores = my_result[0]['scores']
    gt_boxes = gt_result[0]['boxes'][valid_index_gt]
    gt_scores = gt_result[0]['scores'][valid_index_gt]

    loss = 0
    matches = []
    
    # for gt_id, gt_box in enumerate(gt_boxes):
    #     max_iou = 0
    #     max_bid = 0
    #     for my_id, my_box in enumerate(my_boxes):
    #         IoU = jaccard(my_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
    #         if IoU > max_iou:
    #             max_iou = IoU
    #             max_bid = my_id
    #     if max_iou == 0:
    #         loss -= 1.0
    #     else:
    #         matches.append((gt_box, my_boxes[max_bid]))
    #         loss += min((my_scores[max_bid] - gt_scores[gt_id]),0)
    #         # loss += (my_scores[max_bid] - gt_scores[gt_id])

    for gt_id, gt_box in enumerate(gt_boxes):
        max_conf_score = 0
        max_bid = 0
        for my_id, my_box in enumerate(my_boxes):
            IoU = jaccard(my_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
            if IoU > IoU_threshold and my_scores[my_id] > max_conf_score:
                max_conf_score, max_bid = my_scores[my_id], my_id
        if max_conf_score == 0:
            matches.append((gt_id, "NAN"))
            loss -= 1.0
        else:
            matches.append((gt_id, max_bid))
            loss += min((my_scores[max_bid] - gt_scores[gt_id]),0)
            # loss += (my_scores[max_bid] - gt_scores[gt_id])
    
    # return loss, matches
    return loss / len(gt_boxes), matches

def compute_sum_conf_scores(gt_result, car_only = False):
    threshold = 0.5
    if car_only:
        valid_index_gt = (gt_result[0]['scores'] > threshold) & (gt_result[0]['labels'] == 3)
    else:
        valid_index_gt = (gt_result[0]['scores'] > threshold)
    return sum(gt_result[0]['scores'][valid_index_gt])

def compute_saliency(video_slice, inference_result):
    loss = compute_sum_conf_scores(inference_result)
    loss.backward()
    with torch.no_grad():
        saliency = video_slice.grad
    return saliency

def compute_saliency_threshold(saliency, percentile=0.9):
    threshold_list = []
    saliency_reshaped = saliency.clone().detach().numpy().reshape([720*1280, 3])
    saliency_reshaped = abs(saliency_reshaped)
    saliency_r, saliency_g, saliency_b = saliency_reshaped[:,0], saliency_reshaped[:,1], saliency_reshaped[:,2]
    saliency_r.sort()
    saliency_g.sort()
    saliency_b.sort()
    threshold_list.append(saliency_r[int(len(saliency_r)*percentile)-1])
    threshold_list.append(saliency_g[int(len(saliency_g)*percentile)-1])
    threshold_list.append(saliency_b[int(len(saliency_b)*percentile)-1])
    return threshold_list

def compute_saliency_threshold_percent(saliency, percentage=0.1, sum_flag=True):
    saliency_map = abs(saliency.clone())
    if sum_flag:
        saliency_sum = saliency_map.sum(dim = 1).flatten()
    else:
        saliency_sum = saliency_map.flatten()
    saliency_sum, _ = torch.sort(saliency_sum)
    threshold = saliency_sum[int(len(saliency_sum)*(1 - percentage)) - 1]
    return threshold

def detach_result(results):
    for result in results:
        for key in result:
            result[key] = result[key].cpu().detach()
    return results

def print_match_scores(result, gt_result, matches, car_only = False):
    threshold = 0.5
    if car_only:
        valid_index_gt = (gt_result[0]['scores'] > threshold) & (gt_result[0]['labels'] == 3)
    else:
        valid_index_gt = (gt_result[0]['scores'] > threshold)
    gt_boxes = gt_result[0]['boxes'][valid_index_gt]
    gt_scores = gt_result[0]['scores'][valid_index_gt]
    gt_labels = gt_result[0]['labels'][valid_index_gt]

    my_boxes = result[0]['boxes']
    my_scores = result[0]['scores']
    my_labels = result[0]['labels']

    list_score_matches = []
    for index, (gt_id, my_id) in enumerate(matches):
        if my_id == "NAN":
            list_score_matches.append(("NAN", round(gt_scores[gt_id].item(), 3)))
        else:
            list_score_matches.append((round(my_scores[my_id].item(),3), round(gt_scores[gt_id].item(),3), \
                                        round(my_scores[my_id].item() - gt_scores[gt_id].item(),3)))
    pprint.pprint(list_score_matches)

def visualize_bboxes(result, dst_path, img_path=None, img_tensor=None, fid=None, car_only=False):
    # image = cv2.imread("/tank/object_detection_dataset/trafficcam_1_one_frame_2/0000000000.png")
    # print("===== generating result visualizations =====")
    if img_tensor is None:
        image = cv2.imread(img_path)
    else:
        save_tensor_as_image(img_tensor[0], "tmp_intermediate.png")
        image = cv2.imread("tmp_intermediate.png")
    img = image
    # img = (image * 255).astype(np.uint8)

    threshold = 0.5
    if car_only:
        valid_index_mine = (result[0]['scores'] > threshold) & (result[0]['labels'] == 3)
    else:
        valid_index_mine = (result[0]['scores'] > threshold)
    my_boxes = result[0]['boxes'][valid_index_mine]
    my_scores = result[0]['scores'][valid_index_mine]
    my_labels = result[0]['labels'][valid_index_mine]

    for i in range(len(my_boxes)):
        x1, x2, x3, x4 = map(int, my_boxes[i].tolist())
        #print(x1, x2, x3, x4)
        final_image = cv2.rectangle(img, (x1, x2), (x3, x4), (36, 255, 12), 1)
        # random_noise = " " * int(random.random() * 10)
        # cv2.putText(final_image, f"{round(my_scores[i].item(), 3)}{random_noise}{my_labels[i].item()}", \
        #     (x1, x2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        cv2.putText(final_image, f"{round(my_scores[i].item(), 3)}", \
            (x1, x2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

    # visualize gt boxes
    # valid_index_gt = (gt_result[0]['scores'] > threshold) & (gt_result[0]['labels'] == 3)
    # gt_boxes = gt_result[0]['boxes'][valid_index_gt]
    # gt_scores = gt_result[0]['scores'][valid_index_gt]
    # gt_labels = gt_result[0]['labels'][valid_index_gt]
    # for i in range(len(gt_boxes)):
    #     # if result[0]['scores'][i].item() > 0.5:
    #     x1, x2, x3, x4 = map(int, gt_boxes[i].tolist())
    #     #print(x1, x2, x3, x4)
    #     image = cv2.rectangle(img, (x1, x2), (x3, x4), (36, 12, 255), 1)
    #     cv2.putText(image, str(round(gt_scores[i].item(), 3)), (x1, x2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 12, 255), 2)

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    if not fid:
        fname = f'{dst_path}/0000000000.png'
    else:
        fname = f'{dst_path}/{str(fid).zfill(10)}.png'
    cv2.imwrite(fname, final_image)
    
def visualize_bbox_matches(result, gt_result, matches, dst_path, img_path=None, img_tensor=None, car_only=False):
    print("===== generating result visualizations =====")
    if img_tensor is None:
        image = cv2.imread(img_path)
    else:
        save_tensor_as_image(img_tensor[0], "tmp_intermediate.png")
        image = cv2.imread("tmp_intermediate.png")
    img = image
    
    # list of distinct colors
    list_color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),
                    (128,0,0),(128,128,0),(0,128,0),(128,0,128),(0,128,128),(0,0,128)]

    threshold = 0.5
    if car_only:
        valid_index_gt = (gt_result[0]['scores'] > threshold) & (gt_result[0]['labels'] == 3)
    else:
        valid_index_gt = (gt_result[0]['scores'] > threshold)
    gt_boxes = gt_result[0]['boxes'][valid_index_gt]
    gt_scores = gt_result[0]['scores'][valid_index_gt]
    gt_labels = gt_result[0]['labels'][valid_index_gt]

    my_boxes = result[0]['boxes']
    my_scores = result[0]['scores']
    my_labels = result[0]['labels']

    for index, (gt_id, my_id) in enumerate(matches):
        if my_id == "NAN":
            color = (192,192,192)
            gt_box_x1, gt_box_x2, gt_box_x3, gt_box_x4 = map(int, gt_boxes[gt_id].tolist())
            final_image2 = cv2.rectangle(img, (gt_box_x1, gt_box_x2), (gt_box_x3, gt_box_x4), color, 1)
            # random_noise = " " * int(random.random() * 10)
            # cv2.putText(final_image, f"{round(my_scores[i].item(), 3)}{random_noise}{my_labels[i].item()}", \
            #     (x1, x2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            cv2.putText(final_image2, f"{round(gt_scores[gt_id].item(), 3)} {gt_labels[gt_id]}", \
                (gt_box_x1, gt_box_x2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        else:
            color = list_color[index % len(list_color)]

            my_box_x1, my_box_x2, my_box_x3, my_box_x4 = map(int, my_boxes[my_id].tolist())
            final_image = cv2.rectangle(img, (my_box_x1, my_box_x2), (my_box_x3, my_box_x4), color, 1)
            # random_noise = " " * int(random.random() * 10)
            # cv2.putText(final_image, f"{round(my_scores[i].item(), 3)}{random_noise}{my_labels[i].item()}", \
            #     (x1, x2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            cv2.putText(final_image, f"{round(my_scores[my_id].item(), 3)}  {my_labels[my_id]}", \
                (my_box_x1, my_box_x2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            gt_box_x1, gt_box_x2, gt_box_x3, gt_box_x4 = map(int, gt_boxes[gt_id].tolist())
            final_image2 = cv2.rectangle(final_image, (gt_box_x1, gt_box_x2), (gt_box_x3, gt_box_x4), color, 1)
            # random_noise = " " * int(random.random() * 10)
            # cv2.putText(final_image, f"{round(my_scores[i].item(), 3)}{random_noise}{my_labels[i].item()}", \
            #     (x1, x2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            cv2.putText(final_image2, f"{round(gt_scores[gt_id].item(), 3)} {gt_labels[gt_id]}", \
                (gt_box_x1, gt_box_x2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    fname = f'{dst_path}/0000000000.png'
    cv2.imwrite(fname, final_image2)

def tensor_to_RGB_255(video_slice):
    video_slice_pil = T.ToPILImage()(video_slice)
    video_slice_rgb_255 = np.array(video_slice_pil)  # (720, 1280, 3)
    video_slice_rgb_255 = video_slice_rgb_255.astype(np.int32)
    return video_slice_rgb_255

def RGB_255_to_tensor(video_slice_rgb_255):
    return T.ToTensor()(video_slice_rgb_255)

def read_frame(logger, video_path):
    videos, _, _ = read_videos([video_path], logger, normalize=False, from_source=False)
    for fid, video_slice in enumerate(zip(*videos)):
        video_slice = video_slice[0]
    video_slice_255 = tensor_to_RGB_255(video_slice[0])
    return video_slice, video_slice_255

def infer(video_slice, app):
    video_slice.requires_grad = True
    result = app.inference(video_slice.cuda(), nograd=False)
    video_slice = video_slice.detach().cpu()
    return detach_result(result)

def generate_saliency_map(video_slice, app):
    video_slice.requires_grad = True
    result = app.inference(video_slice.cuda(), nograd=False)
    saliency = compute_saliency(video_slice, result)
    # prevent GPU memory from being too full
    video_slice = video_slice.detach().cpu()
    return saliency, result

def save_saliency_maps(list_qps, list_saliency_maps, vid_name):
    saliency_dict = {}
    for qp, saliency_map in zip(list_qps, list_saliency_maps):
        saliency_dict[qp] = saliency_map
    with open(f"pt_archive/{vid_name}_all_saliency_maps.pickle", "wb") as f:
        pickle.dump(saliency_dict, f)

def save_inference_result(list_qps, list_inference_results, vid_name):
    results_dict = {}
    for qp, infer_res in zip(list_qps, list_inference_results):
        results_dict[qp] = infer_res
    with open(f"my_results/{vid_name}_inference_results.pickle", "wb") as f:
        pickle.dump(results_dict, f)

def visualiza_high_saliency_pixels(saliency_mask, video_slice, path="my_vis/avg_saliency.png"):
    video_slice_vis = saliency_mask.unsqueeze(0) * video_slice
    video_slice_255_vis = tensor_to_RGB_255(video_slice_vis[0])
    video_slice_255_vis = video_slice_255_vis.astype(np.uint8)
    img = PIL.Image.fromarray(video_slice_255_vis, 'RGB')
    img.save(path)

def visualize_saliency(saliency_map, img_path, path="saliency_visualization.png"):
    image = PIL.Image.open(img_path)
    saliency_map = torch.sqrt(abs(saliency_map).sum(dim=1, keepdim=True))
    visualize_heat_qz(
        image,
        saliency_map,
        path,
        1
    )

def visualize_saliency_mask(saliency_mask, tile_size, img_path, path="my_vis/avg_saliency_mask.png"):
    # image = PIL.Image.open("/tank/object_detection_dataset/trafficcam_4_one_frame_2/0000000000.png")
    image = PIL.Image.open(img_path)
    visualize_heat_qz(
        image,
        saliency_mask,
        path,
        # 16,
        tile_size
    )
 
def find_isolated_pixels(gt_saliency_mb_density, gt_saliency_mask, lb=30, rb=50):
    list_pixels = []
    gt_saliency_mb_density_mask1 = (gt_saliency_mb_density < rb)
    gt_saliency_mb_density_mask2 = (gt_saliency_mb_density > lb)
    gt_saliency_mb_density_mask = gt_saliency_mb_density_mask1 * gt_saliency_mb_density_mask2
    list_mbs = gt_saliency_mb_density_mask.nonzero()
    for (_, _, i, j) in list_mbs:
        for m in range(i*16, (i+1)*16):
            for n in range(j*16, (j+1)*16):
                if gt_saliency_mask[0][0][m][n] != 0 and gt_saliency_mask[0][0][m][n+1] == 0:
                    list_pixels.append([m,n])
    return list_pixels
    
def visualize_pixel(pixel_index, gt_video_slice_255):
    gt_video_slice_255_vis = copy.deepcopy(gt_video_slice_255)
    gt_video_slice_255_vis[pixel_index[0]-5:pixel_index[0]+5,pixel_index[1]-5:pixel_index[1]+5,:] = [255,0,0]
    gt_video_slice_255_vis = gt_video_slice_255_vis.astype(np.uint8)
    img = PIL.Image.fromarray(gt_video_slice_255_vis, 'RGB')
    img.save("avg_saliency_pixel_vis.png")

def print_associated_diff_values(pixel_index, list_saliency_maps):
    print("current pixel saliency values: ")
    pprint.pprint([i[0,:,pixel_index[0],pixel_index[1]] for i in list_saliency_maps])
    print("adjacent pixel saliency values: ")
    pprint.pprint([i[0,:,pixel_index[0],pixel_index[1]+1] for i in list_saliency_maps])

def plot_cdf(arrs, path="my_plots/cdf.png"):
    for arr in arrs:
        arr.sort()
    indices = [i/len(arrs[0]) for i in range(len(arrs[0]))]
    fig, ax = plt.subplots()
    ax.plot(arrs[0], indices, color='red', label='low saliency in mb')
    ax.plot(arrs[1], indices, color='blue', label='high saliency not in mb')
    ax.legend()
    plt.xlabel("Saliency value")
    plt.title("CDFs of two selected types of pixels")
    plt.savefig(path)

def save_and_encode(hybrid_img, video_name, segment_name, high_quality_qp=1):
    if not os.path.exists(f"{video_name}_{segment_name}"):
        os.mkdir(f"{video_name}_{segment_name}")
    save_tensor_as_image(hybrid_img, path=f"{video_name}_{segment_name}/0000000000.png")
    encode_image(f"{video_name}_{segment_name}", f"{video_name}_{segment_name}.mp4", qp=high_quality_qp)
    return os.path.getsize(f"{video_name}_{segment_name}.mp4")

def save_tensor_as_image(video_slice, path):
    video_slice_255 = tensor_to_RGB_255(video_slice)
    video_slice_255 = video_slice_255.astype(np.uint8)
    video_slice_255 = cv2.cvtColor(video_slice_255, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, video_slice_255)

def encode_image(folder, output_name, qp):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{folder}/%010d.png",
            "-start_number",
            "0",
            "-qp",
            f"{qp}",
            "-vcodec", 
            "libx264",
            "-pix_fmt", 
            "yuv420p",
            output_name,
            "-loglevel",
            "quiet"
        ]
    )


def encode_video(folder, output_name, qp):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{folder}/%010d.png",
            "-start_number",
            "0",
            "-qmin",
            f"{qp}",
            "-qmax",
            f"{qp}",
            "-vcodec", 
            "libx264",
            "-pix_fmt", 
            "yuv420p",
            "-loglevel",
            "quiet",
            output_name
        ]
    )

def store_qp_assignment(vid_name, mask, low_qp=1, high_qp=40, file_name=None):
    if not file_name:
        f = open(f"qp_files/{vid_name}_qp_assignment", "w+")
    else:
        f = open(file_name, "a+")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j].item() == 1.0:
                f.write(f"{low_qp} ")
            else:
                f.write(f"{high_qp} ")
        f.write("\n")

def store_qp_assignment_multi_level(vid_name, mask, file_name=None):
    if not file_name:
        f = open(f"qp_files/{vid_name}_qp_assignment", "w+")
    else:
        f = open(file_name, "a+")
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            f.write(f"{int(mask[i][j].item())} ")
        f.write("\n")

def generate_box_based_mask(result, threshold = 0.6, car_only=False):
    if car_only:
        valid_index = (result[0]['scores'] > threshold) & (result[0]['labels'] == 3)
    else:
        valid_index = (result[0]['scores'] > threshold)
    bboxes = result[0]['boxes'][valid_index].tolist()
    mask = torch.zeros([45,80])

    for x1, y1, x2, y2 in bboxes:
        start_mb_x, start_mb_y = int(x1 // 16), int(y1 // 16)
        end_mb_x, end_mb_y = min(int(x2 // 16)+1, 79), min(int(y2 // 16)+1, 44)
        mask[start_mb_y:end_mb_y+1, start_mb_x:end_mb_x+1] = torch.ones([end_mb_y-start_mb_y+1, end_mb_x-start_mb_x+1])

    return mask

def determine_splits(saliency, qp_levels = 10):
    saliency_map = abs(saliency.clone()).flatten()
    saliency_map, _ = torch.sort(saliency_map, descending=True)
    splits = [0] * qp_levels
    for i in range(qp_levels):
        splits[i] = saliency_map[int(3600/qp_levels*(i+1)-1)].item()
    return splits

def generate_qp_assignment_ten_levels(saliency, low_quality_qp):
    qp_levels = 10
    qp_mask = torch.zeros([45,80])
    qp_list = [1] + [0] * 4 + [low_quality_qp] * 5
    for i in range(1, 5):
        qp_list[i] = qp_list[i-1] + round((low_quality_qp - 1)/5)
    splits = determine_splits(saliency, 10)
    quota = [0] * qp_levels
    for i in range(45):
        for j in range(80):
            for k in range(qp_levels):
                if quota[k] >= int(3600 / qp_levels):
                    continue
                if abs(saliency[i][j]) >= splits[k]:
                    qp_mask[i,j] = qp_list[k]
                    quota[k] += 1
                    break
    return qp_mask, qp_list

def generate_qp_assignment(saliency, avg_qp, qp_levels = 10):
    qp_mask = torch.zeros([45,80])
    # qp_list = [max(1, 2*avg_qp-51)] + [0] * (qp_levels-2) + [min(51, 2*avg_qp-1)]
    qp_list = [1, max(2, 2*avg_qp-51)] + [0] * (qp_levels-4) + [min(50, 2*avg_qp-1), 51]
    for i in range(2,qp_levels-1):
        qp_list[i] = qp_list[i-1] + round((qp_list[-2] - qp_list[1])/(qp_levels - 2))

    splits = determine_splits(saliency, qp_levels)
    quota = [0] * qp_levels
    for i in range(45):
        for j in range(80):
            for k in range(qp_levels):
                if quota[k] >= int(3600 / qp_levels):
                    continue
                if abs(saliency[i][j]) >= splits[k]:
                    qp_mask[i,j] = qp_list[k]
                    quota[k] += 1
                    break

    return qp_mask, qp_list

def write_qp_matrix_file(qp_matrix):
    with open("/tank/qizheng/codec_project/x264/qp_matrix_file", "w+") as f:
        for i in range(45):
            for j in range(80):
                f.write(f"{int(qp_matrix[i][j].item())} ")
            f.write("\n")

def roi_encoding(folder, output_name, qp_matrix):
    write_qp_matrix_file(qp_matrix)

    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = "/tank/qizheng/lib/"

    subprocess.run(
        [
            "/tank/qizheng/codec_project/x264/ffmpeg-3.4.8/ffmpeg",
            "-y",
            "-i",
            f"{folder}/%010d.png",
            "-start_number",
            "0",
            "-qp",
            "1",
            "-vcodec", 
            "libx264",
            "-pix_fmt", 
            "yuv420p",
            "-loglevel",
            "quiet",
            output_name
            # "-loglevel",
            # "quiet"
        ],
        env=ffmpeg_env
    )

def roi_encoding_one_video(folder, output_name, qp_matrix_file_path):
    # overwrite qp_matrix_file
    subprocess.run(
        [
            "cp",
            f"{qp_matrix_file_path}",
            "/tank/qizheng/codec_project/x264/qp_matrix_file"
        ]
    )

    # roi encoding
    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = "/tank/qizheng/lib/"

    subprocess.run(
        [
            "/tank/qizheng/codec_project/x264/ffmpeg-3.4.8/ffmpeg",
            "-y",
            "-i",
            f"{folder}/%010d.png",
            "-start_number",
            "0",
            "-qp",
            "1",
            "-vcodec", 
            "libx264",
            "-pix_fmt", 
            "yuv420p",
            "-loglevel",
            "quiet",
            output_name
            # "-loglevel",
            # "quiet"
        ],
        env=ffmpeg_env
    )

def roi_encoding_one_frame(folder, output_name):

    # roi encoding
    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = "/tank/qizheng/lib/"

    subprocess.run(
        [
            "/tank/qizheng/codec_project/x264/ffmpeg-3.4.8/ffmpeg",
            "-y",
            "-i",
            f"{folder}/%010d.png",
            "-start_number",
            "0",
            "-qp",
            "1",
            "-vcodec", 
            "libx264",
            "-pix_fmt", 
            "yuv420p",
            "-loglevel",
            "quiet",
            "-threads",
            "1",
            output_name
        ],
        env=ffmpeg_env
    )

def roi_encoding_gop(folder, mask, output_name):
    # overwrite qp_matrix_file
    with open("/tank/qizheng/codec_project/x264/qp_matrix_file", "w+") as f:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                f.write(f"{int(mask[i][j])} ")
            f.write("\n")

    # roi encoding
    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = "/tank/qizheng/lib/"

    subprocess.run(
        [
            "/tank/qizheng/codec_project/x264/ffmpeg-3.4.8/ffmpeg",
            "-y",
            "-i",
            f"{folder}/%010d.png",
            "-start_number",
            "0",
            "-qp",
            "1",
            "-vcodec", 
            "libx264",
            "-pix_fmt", 
            "yuv420p",
            "-threads",
            "1",
            "-loglevel",
            "quiet",
            output_name
        ],
        env=ffmpeg_env
    )

def decode_video(video_path, dst_folder_path=None):
    if not dst_folder_path:
        video_frames_path = video_path.split(".")[0]
    else: 
        video_frames_path = dst_folder_path
    if not os.path.exists(video_frames_path):
        os.mkdir(video_frames_path)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{video_path}",
            "-start_number",
            "0",
            "-loglevel",
            "quiet",
            f"{video_frames_path}/%010d.png"
        ]
    )

def get_image(path):
    image = PIL.Image.open(path + "0000000000.png").convert("RGB")

    transform_hq = T.Compose(
        [
            T.Resize((720, 1280)),
            T.ToTensor(),
        ]
    )

    return transform_hq(image)