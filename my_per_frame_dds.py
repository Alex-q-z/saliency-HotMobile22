"""
    Compress the video through gradient-based optimization.
"""

import argparse
import gc
import logging
import time
from pathlib import Path
import pickle
import shutil

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
# working_dir_path = "/datamirror/qizhengz/codec_project/"
working_dir_path = "/tank/qizheng/codec_project/"


def compress_dds(video_input, high_quality_qp, low_quality_qp, img_w = 1280, img_h = 720):

    # initialize
    logger = logging.getLogger("dds")
    logger.addHandler(logging.FileHandler("dds.log"))
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    if img_h % 16 == 0 and img_w % 16 == 0:
        tile_size = 16
    else:
        tile_size = 8
    conf = 0.7
    size_thres = 0.02

    # read the video frames (will use the largest video as ground truth)
    print(f"Reading in video {video_input} for compression (DDS), hqp = {high_quality_qp}, lqp = {low_quality_qp}")

    if not os.path.exists(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/"):
        encode_video(working_dir_path + f"{video_input}", \
                     working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec.mp4", \
                     low_quality_qp)
        decode_video(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec.mp4", \
                     working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/")

    # dds_low_quality_frames = Video_Dataset(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/")
    # # dds_high_quality_frames = Video_Dataset(working_dir_path + f"{video_input}_qp{high_quality_qp}_local_codec/")
    # dds_original_frames = Video_Dataset(working_dir_path + f"{video_input}/")

    dds_original_frame = get_image(working_dir_path + f"{video_input}/").unsqueeze(0)
    dds_low_quality_frame = get_image(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/").unsqueeze(0)

    # construct the mask
    mask_shape = [
        1,
        1,
        img_h // tile_size,
        img_w // tile_size,
    ]
    mask = torch.ones(mask_shape).float()

    # if fid % 100 == 0:
    #     print(f"Processing frame {fid}")

    mask_slice = mask[0:1, :, :, :]

    # dds_original_frame = dds_original_frames[fid]['image'].unsqueeze(0)
    # dds_low_quality_frame = dds_low_quality_frames[fid]['image'].unsqueeze(0)
    # dds_high_quality_frame = dds_high_quality_frames[fid]['image'].unsqueeze(0)

    # first iteration
    lq_inference = app.inference(dds_low_quality_frame.cuda(), detach=True)
    # lq_inference = app.filter_result(lq_inference, None, gt=False, confidence_check=False)
    lq_inference = app.filter_results(detach_result(lq_inference)[0], 0.5) # ???

    proposals = app.region_proposal(dds_low_quality_frame.cuda())
    # proposal_boxes
    # objectness_logits
    proposals = proposals[proposals.objectness_logits > conf]
    proposals = proposals[
        proposals.proposal_boxes.area() < size_thres * img_w * img_h
    ]

    # filter out regions that overlaps with inference results
    iou = pairwise_iou(proposals.proposal_boxes, Boxes(lq_inference[2]))
    iou = iou > 0.3
    iou = iou.sum(dim=1)
    proposals = proposals[iou == 0]
    regions = center_size(proposals.proposal_boxes.tensor).cpu()

    # second iteration
    maskB = generate_mask_from_regions(
        mask_slice.cuda(), regions, 0, tile_size, cuda=True
    )
    mask_delta = maskB
    mask_delta[mask_delta < 0] = 0
    mask_slice[:, :, :, :] = mask_delta

    # mask
    mask.requires_grad = False
    mask_tiled = tile_mask(mask, tile_size)[0][0].unsqueeze(0).unsqueeze(0)

    # save mask
    with open(f"masks/{video_input}_dds_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "wb") as f:
        pickle.dump(mask_tiled, f)

    # encode high-quality regions
    dds_high_quality_region = dds_original_frame.cpu() * mask_tiled.cpu()
    save_and_encode(dds_high_quality_region[0], working_dir_path + video_input, f"dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}", high_quality_qp)

    # encode low-quality background
    # dds_low_quality_region = dds_original_frame.cpu() * (torch.ones_like(mask_tiled) - mask_tiled).cpu()
    save_and_encode(dds_low_quality_frame[0], working_dir_path + video_input, f"dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}", low_quality_qp)

    # encoding for smoothing (no need now)
    # mask_tiled_reuse = None
    # with open(f"masks/videos/trafficcam_2_{int((frame_idx - 1) // 10 * 10 + 1)}_dds_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "rb") as f:
    #     mask_tiled_reuse = pickle.load(f)

    # dds_high_quality_region_smooth = dds_original_frame.cpu() * mask_tiled_reuse.cpu()
    # save_and_encode(dds_high_quality_region_smooth[0], video_input, f"dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth", high_quality_qp)

    # dds_low_quality_region_smooth = dds_original_frame.cpu() * (torch.ones_like(mask_tiled_reuse) - mask_tiled_reuse).cpu()
    # save_and_encode(dds_low_quality_region_smooth[0], video_input, f"dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth", low_quality_qp)


def my_inference(video_input, high_quality_qp, low_quality_qp, stats_file_name):
    
    # read in the encoded video as Dataset objects
    print(f"Reading in video {video_input} for inference")
    # gt_frames = Video_Dataset(working_dir_path + f"{video_input}/")
    video_slice_gt = get_image(working_dir_path + f"{video_input}/").unsqueeze(0)

    # non-smoothing
    decode_video(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}.mp4",
                 working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_decoded",)
    # dds_high_quality_frames = Video_Dataset(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_decoded/")
    dds_high_quality_frame = get_image(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_decoded/").unsqueeze(0)

    decode_video(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}.mp4",
                 working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_decoded",)
    # dds_low_quality_frames = Video_Dataset(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_decoded/")
    dds_low_quality_frame = get_image(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_decoded/").unsqueeze(0)

    with open(f"masks/{video_input}_dds_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "rb") as f:
        mask = pickle.load(f)
    
    # smoothing (no need now)
    # decode_video(video_input + f"_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth.mp4",
    #              video_input + f"_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth_decoded",)
    # dds_high_quality_frames_smooth = Video_Dataset(f"{video_input}" + f"_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth_decoded/")
    
    # decode_video(video_input + f"_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth.mp4",
    #              video_input + f"_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth_decoded",)
    # dds_low_quality_frames_smooth = Video_Dataset(f"{video_input}" + f"_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth_decoded/")

    # with open(f"masks/videos/trafficcam_2_{int((frame_idx - 1) // 10 * 10 + 1)}_dds_per_frame_{high_quality_qp}_{low_quality_qp}.pt", "rb") as f:
    #     mask_reuse = pickle.load(f)

    loss_dds, loss_dds_smooth = 0, 0

    # process the video frame by frame

    # if fid % 100 == 0:
    #     print(f"Processing frame {fid}")

    # compute gt result and qp 40 result for this frame
    # video_slice_gt = gt_frames[fid]['image'].unsqueeze(0)
    gt_result = infer(video_slice_gt, app)

    # non-smoothing
    # dds_high_quality_frame = dds_high_quality_frames[fid]['image'].unsqueeze(0)
    # dds_low_quality_frame = dds_low_quality_frames[fid]['image'].unsqueeze(0)
    dds_hybrid_frame = dds_high_quality_frame.cpu() * mask.cpu() + \
                        dds_low_quality_frame.cpu() * (torch.ones_like(mask) - mask).cpu()
    dds_hybrid_frame.require_grad = True
    result_dds_hybrid_frame = app.inference(dds_hybrid_frame.cuda(), nograd=False)
    result_dds_hybrid_frame = detach_result(result_dds_hybrid_frame)
    loss_dds_hybrid_frame, _ = compute_loss(result_dds_hybrid_frame, gt_result)
    loss_dds += abs(loss_dds_hybrid_frame)

    # smoothing (no need now)
    # dds_high_quality_frame_smooth = dds_high_quality_frames_smooth[fid]['image'].unsqueeze(0)
    # dds_low_quality_frame_smooth = dds_low_quality_frames_smooth[fid]['image'].unsqueeze(0)
    # dds_hybrid_frame_smooth = dds_high_quality_frame_smooth.cpu() * mask_reuse.cpu() + \
    #                     dds_low_quality_frame_smooth.cpu() * (torch.ones_like(mask_reuse) - mask_reuse).cpu()
    # dds_hybrid_frame_smooth.require_grad = True
    # result_dds_hybrid_frame_smooth = app.inference(dds_hybrid_frame_smooth.cuda(), nograd=False)
    # result_dds_hybrid_frame_smooth = detach_result(result_dds_hybrid_frame_smooth)
    # loss_dds_hybrid_frame_smooth, _ = compute_loss(result_dds_hybrid_frame_smooth, gt_result)
    # loss_dds_smooth += abs(loss_dds_hybrid_frame_smooth)

    # compute average loss per frame
    # loss_dds /= len(gt_frames)
    # loss_dds_smooth /= len(gt_frames)

    # get file sizes
    dds_high_quality_frames_fs = os.path.getsize(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}.mp4")
    dds_low_quality_frames_fs = os.path.getsize(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}.mp4")
    dds_fs = dds_high_quality_frames_fs + dds_low_quality_frames_fs

    # dds_high_quality_frames_fs_smooth = os.path.getsize(f"{video_input}" + f"_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_smooth.mp4")
    # dds_low_quality_frames_fs_smooth = os.path.getsize(f"{video_input}" + f"_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_smooth.mp4")
    # dds_fs_smooth = dds_high_quality_frames_fs_smooth + dds_low_quality_frames_fs_smooth

    # with open(f"stats_files/tradeoff_files/tf2_1800_frames_dds_per_frame_es", "a+") as f:
    #     if not isinstance(loss_dds, float) and not isinstance(loss_dds_smooth, float):
    #         f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_dds).item()},{dds_fs},{abs(loss_dds_smooth).item()},{dds_fs_smooth}\n")
    #     elif not isinstance(loss_dds_smooth, float):
    #         f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_dds)},{dds_fs},{abs(loss_dds_smooth).item()},{dds_fs_smooth}\n")
    #     else:
    #         f.write(f"{video_input},{fid},{high_quality_qp},{low_quality_qp},{abs(loss_dds)},{dds_fs},{abs(loss_dds_smooth)},{dds_fs_smooth}\n")

    # with open(f"stats_files/tradeoff_files/tf2_first_300_frames_dds_per_frame_1101", "a+") as f:
    with open(stats_file_name, "a+") as f:
        if not isinstance(loss_dds, float):
            f.write(f"{video_input},0,{high_quality_qp},{low_quality_qp},{abs(loss_dds).item()},{dds_fs}\n")
        else:
            f.write(f"{video_input},0,{high_quality_qp},{low_quality_qp},{abs(loss_dds)},{dds_fs}\n")

    # cleaning for saving disk space (optional)
    if os.path.exists(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec.mp4"):
        os.remove(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec.mp4")
    # shutil.rmtree(working_dir_path + f"{video_input}_qp{low_quality_qp}_local_codec/")

    os.remove(f"masks/{video_input}_dds_per_frame_{high_quality_qp}_{low_quality_qp}.pt")
    os.remove(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}.mp4")
    os.remove(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}.mp4")
    shutil.rmtree(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}_decoded/", ignore_errors=True)
    shutil.rmtree(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}_decoded/", ignore_errors=True)
    shutil.rmtree(working_dir_path + f"{video_input}_dds_per_frame_hq_{high_quality_qp}_{low_quality_qp}/")
    shutil.rmtree(working_dir_path + f"{video_input}_dds_per_frame_lq_{high_quality_qp}_{low_quality_qp}/")


##### main function
# video_list = [f"videos/trafficcam_2_{i}" for i in range(1,1801,10)]
# video_list = [f"videos/trafficcam_2_{i}" for i in [331]]
# vid_names = ["videos/tf1"] * 300 + ["videos/tf2"] * 300 + ["videos/tf3"] * 300 + ["videos/tf4"] * 300 + ["videos/tf5"] * 300
# vid_names = ["videos/tf4"] * 300 + ["videos/tf5"] * 300
# video_list = [f"videos/trafficcam_1_{i}" for i in range(1,301,1)] + \
#              [f"videos/trafficcam_2_{i}" for i in range(1,301,1)] + \
#              [f"videos/trafficcam_3_{i}" for i in range(1,301,1)] + \
#              [f"videos/trafficcam_4_{i}" for i in range(1,301,1)] + \
#              [f"videos/trafficcam_5_{i}" for i in range(1,301,1)]
# video_list = [f"videos/trafficcam_4_{i}" for i in range(1,301,1)] + \
#              [f"videos/trafficcam_5_{i}" for i in range(1,301,1)]

# HotMobile submiited experiments
# vid_names = ["videos/tf1"] * 300
# video_list = [f"videos/trafficcam_1_{i}" for i in range(1,301,1)]

# Camera-ready experiments (diverse scenes)
# resolutions = [(1280, 720)] * 300 + \
#               [(1920, 1080)] * 600
# vid_names = ["videos/cityscape_6"] * 300 + \
#             ["videos/cityscape_7"] * 300 + \
#             ["videos/cityscape_8"] * 300
# video_list = [f"videos/cityscape_6_{i}" for i in range(1,301,1)] + \
#              [f"videos/cityscape_7_{i}" for i in range(1,301,1)] + \
#              [f"videos/cityscape_8_{i}" for i in range(1,301,1)]
resolutions = [(1920, 1080)] * 1500
vid_names = ["videos/cityscape_1"] * 300 + \
            ["videos/cityscape_2"] * 300 + \
            ["videos/cityscape_4"] * 300 + \
            ["videos/cityscape_7"] * 300 + \
            ["videos/cityscape_8"] * 300
video_list = [f"videos/cityscape_1_{i}" for i in range(1,301,1)] + \
             [f"videos/cityscape_2_{i}" for i in range(1,301,1)] + \
             [f"videos/cityscape_4_{i}" for i in range(1,301,1)] + \
             [f"videos/cityscape_7_{i}" for i in range(1,301,1)] + \
             [f"videos/cityscape_8_{i}" for i in range(1,301,1)]

# hq_qps = [0, 10, 20]
# lq_qps = [40, 40, 40]
# hq_qps = [2] * 6 + [6] * 6 + [10] * 6 + [14] * 6
# lq_qps = [28, 30, 32, 34, 36, 38]*4
# hq_qps = [2] * 3
hq_qps = [2]
# lq_qps = [28, 30, 32, 34, 36, 38]*4
# lq_qps = [28, 32, 36]
lq_qps = [36]

time_compression, time_inference = 0.0, 0.0

# compress dds
for frame_index in range(len(video_list)):
    video = video_list[frame_index]
    vid_name = vid_names[frame_index]
    resolution = resolutions[frame_index]

    # file creation
    # print(f"video name: {video}")
    frame_index_in_video = int(video.split("_")[-1]) - 1
    # import pdb; pdb.set_trace()
    if not os.path.exists(working_dir_path + video):
        os.mkdir(working_dir_path + video)
        subprocess.run(["cp", f"{working_dir_path}{vid_name}/{str(frame_index_in_video).zfill(10)}.png", \
                        f"{working_dir_path}{video}/0000000000.png"])
    # import pdb; pdb.set_trace()
    
    for i in range(len(hq_qps)):
        hq_qp, lq_qp = hq_qps[i], lq_qps[i]

        stats_file_name = f"stats_files_cr/{vid_name}_300_frames_dds_{hq_qp}_{lq_qp}_cr"

        # compression
        start_compress = time.time()
        compress_dds(video, hq_qp, lq_qp, resolution[0], resolution[1])
        end_compress = time.time()
        time_compression += (end_compress - start_compress)

        # inference
        start_infer = time.time()
        my_inference(video, hq_qp, lq_qp, stats_file_name)
        end_infer = time.time()
        time_inference += (end_infer - start_infer)

print(time_compression)
print(time_inference)