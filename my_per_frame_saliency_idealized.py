"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
import time
from pathlib import Path
import pickle

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from detectron2.structures.boxes import pairwise_iou
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dnn.dnn_factory import DNN_Factory
from utils.bbox_utils import center_size
from utils.loss_utils import focal_loss as get_loss
from utils.mask_utils import *
from utils.results_utils import read_ground_truth, read_results
from utils.timer import Timer
from utils.video_utils import get_qp_from_name, read_videos, write_video
from utils.visualize_utils import visualize_heat_by_summarywriter

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

from my_utils import *

# construct applications
dnn = "FasterRCNN_ResNet50_FPN"
# dnn = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
app = DNN_Factory().get_model(dnn)
app.cuda()

# working_dir_path = "/tank/qizheng/codec_project/accmpeg/"
working_dir_path = "/datamirror/qizhengz/codec_project/"


def compress_saliency(video_input, high_quality_qp, low_quality_qp):

    # initialize
    logger = logging.getLogger("saliency")
    logger.addHandler(logging.FileHandler("saliency.log"))

    percent = 0.1

    # read the video frames (will use the largest video as ground truth)
    print(f"Reading in video {video_input} for compression (saliency-based), hqp = {high_quality_qp}, lqp = {low_quality_qp}")

    if not os.path.exists(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/"):
        encode_video(working_dir_path + f"{video_input}", \
                     working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec.mp4", \
                     low_quality_qp)
        decode_video(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec.mp4", \
                     working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/")

    saliency_original_frames = Video_Dataset(working_dir_path + f"{video_input}/")
    saliency_lq_frames = Video_Dataset(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/")
    
    for fid in range(len(saliency_original_frames)):

        # if fid % 100 == 0:
        #     print(f"Processing frame {fid}")

        video_slice_gt = saliency_original_frames[fid]['image'].unsqueeze(0)
        video_slice_lq = saliency_lq_frames[fid]['image'].unsqueeze(0)
        smap, _ = generate_saliency_map(video_slice_gt, app) # saliency computation
        # smap, _ = generate_saliency_map(video_slice_lq, app) # saliency computation
        emap = abs(video_slice_lq - video_slice_gt)
        saliency_sum = abs(smap * emap)

        # compute mb-level saliency mask
        saliency_sum_mb = F.conv2d(
            saliency_sum.sum(dim=1, keepdim=True), 
            torch.ones([1, 1, 16, 16]),
            stride=16
        )
        threshold_mb = compute_saliency_threshold_percent(saliency_sum_mb, percent, sum_flag = False)
        saliency_mask_mb = (saliency_sum_mb > threshold_mb).float()
        saliency_mask_mb_tiled = tile_mask(saliency_mask_mb, 16)[0][0].unsqueeze(0).unsqueeze(0)
        
    # save mask
    with open(f"masks/{video_input}_saliency_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "wb") as f:
        pickle.dump(saliency_mask_mb_tiled, f)

    # encode low-quality regions
    saliency_low_quality_region = video_slice_gt.cpu() * (torch.ones_like(saliency_mask_mb_tiled) - saliency_mask_mb_tiled).cpu()
    save_and_encode(saliency_low_quality_region[0], working_dir_path + video_input, f"saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}", low_quality_qp)

    # encode high-quality regions
    saliency_high_quality_region = video_slice_gt.cpu() * saliency_mask_mb_tiled.cpu()
    save_and_encode(saliency_high_quality_region[0], working_dir_path + video_input, f"saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}", high_quality_qp)

    # encode low-quality background
    # save_and_encode(video_slice_lq[0], working_dir_path + video_input, f"saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}", low_quality_qp)

    # encoding for smoothing (no need now)
    # mask_tiled_reuse = None
    # with open(f"masks/videos/trafficcam_2_{int((frame_idx - 1) // 10 * 10 + 1)}_saliency_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "rb") as f:
    #     mask_tiled_reuse = pickle.load(f)

    # saliency_high_quality_region_smooth = video_slice_gt.cpu() * mask_tiled_reuse.cpu()
    # save_and_encode(saliency_high_quality_region_smooth[0], video_input, f"saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth", high_quality_qp)

    # saliency_low_quality_region_smooth = video_slice_gt.cpu() * (torch.ones_like(mask_tiled_reuse) - mask_tiled_reuse).cpu()
    # save_and_encode(video_slice_lq[0], video_input, f"saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth", low_quality_qp)


def my_inference(video_input, high_quality_qp, low_quality_qp, stats_file_name):
    
    # read in the encoded video as Dataset objects
    print(f"Reading in video {video_input} for inference")
    gt_frames = Video_Dataset(working_dir_path + f"{video_input}/")

    # non-smoothing
    decode_video(working_dir_path + f"{video_input}_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}.mp4",
                 working_dir_path + f"{video_input}_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_decoded",)
    saliency_high_quality_frames = Video_Dataset(working_dir_path + f"{video_input}_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_decoded/")
    
    decode_video(working_dir_path + f"{video_input}_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}.mp4",
                 working_dir_path + f"{video_input}_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_decoded",)
    saliency_low_quality_frames = Video_Dataset(working_dir_path + f"{video_input}_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_decoded/")

    with open(f"masks/{video_input}_saliency_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "rb") as f:
        mask = pickle.load(f)
    
    # smoothing (no need now)
    # decode_video(video_input + f"_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth.mp4",
    #              video_input + f"_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth_decoded",)
    # saliency_high_quality_frames_smooth = Video_Dataset(f"{video_input}" + f"_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth_decoded/")
    
    # decode_video(video_input + f"_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth.mp4",
    #              video_input + f"_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth_decoded",)
    # saliency_low_quality_frames_smooth = Video_Dataset(f"{video_input}" + f"_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth_decoded/")

    # with open(f"masks/videos/trafficcam_2_{int((frame_idx - 1) // 10 * 10 + 1)}_saliency_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "rb") as f:
    #     mask_reuse = pickle.load(f)

    loss_saliency, loss_saliency_smooth = 0, 0

    # process the video frame by frame
    for fid in range(len(gt_frames)):

        # if fid % 100 == 0:
        #     print(f"Processing frame {fid}")

        # compute gt result and qp 40 result for this frame
        video_slice_gt = gt_frames[fid]['image'].unsqueeze(0)
        gt_result = infer(video_slice_gt, app)

        # non-smoothing
        saliency_high_quality_frame = saliency_high_quality_frames[fid]['image'].unsqueeze(0)
        saliency_low_quality_frame = saliency_low_quality_frames[fid]['image'].unsqueeze(0)
        saliency_hybrid_frame = saliency_high_quality_frame.cpu() * mask.cpu() + \
                            saliency_low_quality_frame.cpu() * (torch.ones_like(mask) - mask).cpu()
        saliency_hybrid_frame.require_grad = True
        result_saliency_hybrid_frame = app.inference(saliency_hybrid_frame.cuda(), nograd=False)
        result_saliency_hybrid_frame = detach_result(result_saliency_hybrid_frame)
        loss_saliency_hybrid_frame, _ = compute_loss(result_saliency_hybrid_frame, gt_result)
        loss_saliency += abs(loss_saliency_hybrid_frame)

        # smoothing (no need now)
        # saliency_high_quality_frame_smooth = saliency_high_quality_frames_smooth[fid]['image'].unsqueeze(0)
        # saliency_low_quality_frame_smooth = saliency_low_quality_frames_smooth[fid]['image'].unsqueeze(0)
        # saliency_hybrid_frame_smooth = saliency_high_quality_frame_smooth.cpu() * mask_reuse.cpu() + \
        #                     saliency_low_quality_frame_smooth.cpu() * (torch.ones_like(mask_reuse) - mask_reuse).cpu()
        # saliency_hybrid_frame_smooth.require_grad = True
        # result_saliency_hybrid_frame_smooth = app.inference(saliency_hybrid_frame_smooth.cuda(), nograd=False)
        # result_saliency_hybrid_frame_smooth = detach_result(result_saliency_hybrid_frame_smooth)
        # loss_saliency_hybrid_frame_smooth, _ = compute_loss(result_saliency_hybrid_frame_smooth, gt_result)
        # loss_saliency_smooth += abs(loss_saliency_hybrid_frame_smooth)

    # compute average loss per frame
    loss_saliency /= len(gt_frames)
    # loss_saliency_smooth /= len(gt_frames)

    # get file sizes
    saliency_high_quality_frames_fs = os.path.getsize(working_dir_path + f"{video_input}_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}.mp4")
    saliency_low_quality_frames_fs = os.path.getsize(working_dir_path + f"{video_input}_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}.mp4")
    saliency_fs = saliency_high_quality_frames_fs + saliency_low_quality_frames_fs

    # saliency_high_quality_frames_fs_smooth = os.path.getsize(f"{video_input}" + f"_saliency_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth.mp4")
    # saliency_low_quality_frames_fs_smooth = os.path.getsize(f"{video_input}" + f"_saliency_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth.mp4")
    # saliency_fs_smooth = saliency_high_quality_frames_fs_smooth + saliency_low_quality_frames_fs_smooth

    # with open(f"stats_files/tradeoff_files/tf2_1800_frames_tmp_4", "a+") as f:
    #     if not isinstance(loss_saliency, float) and not isinstance(loss_saliency_smooth, float):
    #         f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_saliency).item()},{saliency_fs},{abs(loss_saliency_smooth).item()},{saliency_fs_smooth}\n")
    #     elif not isinstance(loss_saliency_smooth, float):
    #         f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_saliency)},{saliency_fs},{abs(loss_saliency_smooth).item()},{saliency_fs_smooth}\n")
    #     else:
    #         f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_saliency)},{saliency_fs},{abs(loss_saliency_smooth)},{saliency_fs_smooth}\n")

    # with open(f"stats_files/tradeoff_files/tf2_first_300_frames_saliency_per_frame_idealized_1101", "a+") as f:
    with open(stats_file_name, "a+") as f:
        if not isinstance(loss_saliency, float):
            f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_saliency).item()},{saliency_fs}\n")
        else:
            f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_saliency)},{saliency_fs}\n")

    os.remove(f"masks/{video_input}_saliency_per_frame_{high_quality_qp}_{low_quality_qp}.pt")

##### main function
vid_names = ["videos/tf1"] * 300 + ["videos/tf2"] * 300 + ["videos/tf3"] * 300 + ["videos/tf4"] * 300 + ["videos/tf5"] * 300
video_list = [f"videos/trafficcam_1_{i}" for i in range(1,301,1)] + \
             [f"videos/trafficcam_2_{i}" for i in range(1,301,1)] + \
             [f"videos/trafficcam_3_{i}" for i in range(1,301,1)] + \
             [f"videos/trafficcam_4_{i}" for i in range(1,301,1)] + \
             [f"videos/trafficcam_5_{i}" for i in range(1,301,1)]
# vid_names = ["videos/tf5"] * 1800
# video_list = [f"videos/trafficcam_5_{i}" for i in range(1,1801,1)]

# hq_qps = [0, 10, 20]
# lq_qps = [40, 40, 40]
# hq_qps = [2] * 6 + [6] * 6 + [10] * 6 + [14] * 6
hq_qps = [2] * 3
# lq_qps = [28, 30, 32, 34, 36, 38]*4
lq_qps = [28, 32, 36]

# compress saliency
for i in range(len(video_list)):
    video = video_list[i]
    vid_name = vid_names[i]

    for i in range(len(hq_qps)):
        hq_qp, lq_qp = hq_qps[i], lq_qps[i]

        stats_file_name = f"stats_files/{vid_name}_300_frames_saliency_idealized_{hq_qp}_{lq_qp}"
        
        # compression
        compress_saliency(video, hq_qp, lq_qp)

        # inference
        my_inference(video, hq_qp, lq_qp, stats_file_name)
