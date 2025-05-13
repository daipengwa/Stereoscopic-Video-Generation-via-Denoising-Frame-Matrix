# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2, os, glob
import numpy as np


import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from pypfm import PFMLoader
import imageio, math

import torch.nn.functional as F


DEVICE = 'cuda'
loader = PFMLoader(color=False, compress=False)


def load_frame_depths(depth_root, depth_suffix='.npy', farest_disparity=0.05):

    if depth_suffix == '.npy':
        depth_names = sorted(glob.glob(depth_root + '/*.npy'))
    elif depth_suffix == '.png':
        depth_names = sorted(glob.glob(depth_root + '/*.png'))
    else:
        depth_names = sorted(glob.glob(depth_root + '/*.pfm')) 
  
    num_frames = len(depth_names)
    depths = []
    for i in range(num_frames):
        depth_name = depth_names[i]
        if depth_name.endswith('png'):
            depth = cv2.imread(depth_name, -1)[..., None]  # relative depth, the baseline should be changed scene by scene.
        elif depth_name.endswith('pfm'):
            depth = loader.load_pfm(depth_name)
            depth = np.flipud(depth)[..., None]
        else:
            depth = np.load(depth_name)[..., None]

        depth = np.maximum(depth, 0)  # require >= 0
        depths.append(depth)
    depths = np.stack(depths, axis=0) # NxHxWx1

    return depths

    # # normalize the depth value
    # depths_normlized = (depths - depths.min())/(depths.max() - depths.min())  # normalize to 0~1
    # depths_normlized = depths_normlized + farest_disparity

    # return depths_normlized

def load_frame_rgbs(rgb_root, rgb_suffix='.jpg'):

    rgb_names = sorted(glob.glob(rgb_root + '/*%s' % rgb_suffix))
    num_frames = len(rgb_names)
    rgbs = []

    for i in range(num_frames):
        rgb_name = rgb_names[i]
        rgb = cv2.imread(rgb_name)[:,:,::-1] # bgr2rgb
        rgbs.append(rgb)
    rgbs = np.stack(rgbs, axis=0)

    return rgbs


def warp(x, flo, mode='bilinear'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode=mode)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid, mode=mode)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output, mask
    

def load_image(imfile, t_h=480, t_w=640):
    img = np.array(Image.open(imfile)).astype(np.float32)
    img = cv2.resize(img, (t_w, t_h))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    track_x_all = []
    track_y_all = []
    depths_tracked = []
    depths_refined_tracked = []

    with torch.no_grad():

        # load dataset
        # import pdb; pdb.set_trace()
        rgbs = load_frame_rgbs(args.path)
        depths = load_frame_depths(args.depth_path)
        depths_refined = load_frame_depths(args.depth_path.replace('depths', 'depths_refined'))
        num_frames = rgbs.shape[0]
        
        track_x = args.track_x
        track_y = args.track_y

        for i in range(15):
            cur_id = i
            cur_rgb = rgbs[cur_id]
            cur_depth = depths[cur_id]
            cur_mask = np.ones_like(cur_depth)
            
            ref_id = cur_id + 1
            ref_rgb = rgbs[ref_id]
            ref_depth = depths[ref_id]

            # to tensor
            image1 = torch.from_numpy(cur_rgb[None, ...]).permute(0, 3, 1, 2).float().to(DEVICE)
            image2 = torch.from_numpy(ref_rgb[None, ...]).permute(0, 3, 1, 2).float().to(DEVICE)
            depth2 = torch.from_numpy(np.float32(ref_depth[None, ...])).permute(0,3,1,2).cuda()

            # run raft
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=32, test_mode=True)  # the shape of flow up: tensor[1, 2, h, w]; 2: (f_x, f_y)  image1_x + f_x == image2_x

            print(flow_up[:,0, track_y, track_x])
            track_x = track_x + int(flow_up[:,0, track_y, track_x])
            track_y = track_y + int(flow_up[:,1, track_y, track_x])
            
            track_x_all.append(track_x)
            track_y_all.append(track_y)
            depths_tracked.append(depths[i, track_y, track_x, 0])
            depths_refined_tracked.append(depths_refined[i, track_y, track_x, 0])
            
        print(track_x_all)
        print(track_y_all)
        print(depths_tracked)
        print(depths_refined_tracked)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--path', type=str, default='/usr/local/google/home/daip/Projects/videos/zeroscope/A_knight_riding_on_a_horse_through_the_countryside_320x576/images/', help="color image path")
    parser.add_argument('--depth_path',type=str, default='/usr/local/google/home/daip/Projects/videos/zeroscope/A_knight_riding_on_a_horse_through_the_countryside_320x576/depths/', help="depth image path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--out_dir', type=str, default='//usr/local/google/home/daip/Projects/videos/zeroscope/A_knight_riding_on_a_horse_through_the_countryside_320x576/depths_refined/', help="output dir")
    parser.add_argument('--file_extension', type=str, default='jpg', help="extension of file name")
    parser.add_argument('--window_size', type=int, default=-1, help="window size for temporal smoothness")
    parser.add_argument('--rgb_threshold', type=int, default=20, help="consistency check, avoid misalignment")
    parser.add_argument('--track_x', type=int, default=300, help="")
    parser.add_argument('--track_y', type=int, default=300, help="")
    args = parser.parse_args()

    demo(args)


    