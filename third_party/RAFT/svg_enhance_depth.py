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
            depth = cv2.imread(depth_name, -1)[..., None]  # relative depth
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


def consistency_check(rgbs, masks, depths, rgb_threshold=20):

    '''
    Optical flows are not always correct. use rgb and depth to remove outliers 
    '''
    # rgbs: 0~255 
    # depth: > 0

    masks_new = []
    num_frames = len(rgbs)

    ref_id = int((num_frames-1)/2)
    ref_depth = depths[ref_id]
    ref_rgb = rgbs[ref_id]  

    for i in range(num_frames):
        cur_rgb = rgbs[i]
        cur_depth = depths[i]

        mask_new = masks[i] + 0

        ## rgb_check
        diff_rgb = np.max(np.abs(cur_rgb - ref_rgb), axis=2, keepdims=True)
        mask_new[np.where(diff_rgb>rgb_threshold)] = 0

        ## depth check
        diff_depth = np.abs(cur_depth - ref_depth)
        depth_threshold = 0.5*(cur_depth.max() - cur_depth.min())
        mask_new[np.where(diff_depth>depth_threshold)] = 0
        
        masks_new.append(mask_new)
    
    return masks_new


def temporal_smoothness(rgbs, depths, masks, smooth_kernel):

    num_frames = len(depths)

    depth_after_t = 0
    mask_after_t = 0
    weight_after_t = 0

    for i in range(num_frames):
        depth_after_t = depths[i]*masks[i]*smooth_kernel[i] + depth_after_t
        weight_after_t = masks[i]*smooth_kernel[i] + weight_after_t
        mask_after_t = masks[i] + mask_after_t

    depth_after_t = depth_after_t/(weight_after_t+1e-10)

    return depth_after_t, mask_after_t


def spatial_smoothness(guide_img, input_img, mask, radius_whole=8, radius_non_tracked=8, epsilon=0.0001, iter_whole=1, iter_non_tracked=0):

    mask[np.where(mask<=1)] = 0
    mask[np.where(mask>0)] = 1

    # iterations on whole regions
    filtered_image = input_img + 0
    for i in range(iter_whole):
        filtered_image = cv2.ximgproc.guidedFilter(guide_img, filtered_image, radius_whole, epsilon)
        filtered_image[np.where(filtered_image < 0)] = 0
        filtered_image = filtered_image[..., None]
    
    # apply more iterations on non-tracked regions
    filtered_image_tmp = filtered_image + 0
    for i in range(iter_non_tracked):
        filtered_image = cv2.ximgproc.guidedFilter(guide_img, filtered_image, radius_non_tracked, epsilon)
        filtered_image[np.where(filtered_image < 0)] = 0
        filtered_image = filtered_image[..., None]
        filtered_image = filtered_image * (1-mask) + filtered_image_tmp * mask


    return filtered_image


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        # load dataset
        rgbs = load_frame_rgbs(rgb_root=args.path, rgb_suffix=args.file_suffix)
        depths = load_frame_depths(args.depth_path)
        num_frames = rgbs.shape[0]

        # create temporal smoothness kernel
        window_size = args.window_size  # must be odd
        kernel = np.zeros(window_size)
        mid_idx = (window_size-1)/2
        eps = 1  # control smoothness strength, 0 is the mean filter
        for i in range(window_size):
            kernel[i] = 0.5**(np.abs(i - mid_idx))
        kernel = kernel**eps
        kernel = kernel / np.sum(kernel)  # normalized kernel

        # create output path
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        for i in range(num_frames):

            cur_id = i
            cur_rgb = rgbs[cur_id]
            cur_depth = depths[cur_id]
            cur_mask = np.ones_like(cur_depth)

            warped_rgbs = []
            warped_masks = []
            warped_depths = []

            for j in range(window_size):
                ref_id = cur_id - int((window_size-1)/2) + j

                print('cur_id: %s, ref_id: %s' % (cur_id, ref_id))

                if ref_id < 0 or ref_id >= num_frames:
                    warped_mask = np.zeros_like(cur_mask)  # zeros indicate invalid pixels
                    warped_masks.append(warped_mask)
                    warped_rgbs.append(cur_rgb)
                    warped_depths.append(cur_depth)
                    continue

                if ref_id == cur_id:
                    warped_masks.append(cur_mask)
                    warped_depths.append(cur_depth)
                    warped_rgbs.append(cur_rgb)
                    continue

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
            
                image2_warped, _ = warp(image2, flow_up, mode='nearest')
                depth2_warped, mask2_warped = warp(depth2, flow_up, mode='nearest')
                image2_warped = image2_warped[0, :, :, :].permute(1,2,0).cpu().numpy()
                depth2_warped = depth2_warped[0, :, :, :].permute(1,2,0).cpu().numpy()
                mask2_warped = mask2_warped[0, :, :, :].permute(1,2,0).cpu().numpy()

                warped_rgbs.append(image2_warped)
                warped_depths.append(depth2_warped)
                warped_masks.append(mask2_warped)

            # visualize some intermediate results
            cv2.imwrite('./result/img.png', warped_rgbs[1][:,:,::-1])
            cv2.imwrite('./result/img_warped.png', warped_rgbs[2][:,:,::-1])
            cv2.imwrite('./result/depth.png', np.uint8(255*(warped_depths[1]/warped_depths[1].max())))
            cv2.imwrite('./result/depth_warped.png', np.uint8(255*(warped_depths[2]/warped_depths[1].max())))

            # consistency check
            warped_masks = consistency_check(rgbs=warped_rgbs, masks=warped_masks, depths=warped_depths, rgb_threshold=args.rgb_threshold)

            # temporal smoothness
            depth_after_t, mask_after_t = temporal_smoothness(rgbs=warped_rgbs, depths=warped_depths, masks=warped_masks, smooth_kernel=kernel)

            # spatial smoothness. (guided filter)  
            # default radius_whole is 8. sometimes the guided filter yields nan, try to enlarge the radius_whole.
            depth_after_t_s = spatial_smoothness(guide_img=cur_depth, input_img=depth_after_t, mask=mask_after_t, radius_whole=30, radius_non_tracked=8, epsilon=0.0001, iter_whole=1, iter_non_tracked=0)

            np.save(args.out_dir + '/%05d_depth.npy' % cur_id, depth_after_t_s[:,:,0])
            cv2.imwrite(args.out_dir + '/%05d_depth.png' % cur_id, np.uint8(np.minimum(depth_after_t_s, 255)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--path', type=str, default='/Videos/zeroscope/A_knight_riding_on_a_horse_through_the_countryside_320x576/images/', help="color image path")
    parser.add_argument('--depth_path',type=str, default='/Videos/zeroscope/A_knight_riding_on_a_horse_through_the_countryside_320x576/depths/', help="depth image path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--out_dir', type=str, default='/Videos/zeroscope/A_knight_riding_on_a_horse_through_the_countryside_320x576/depths_refined/', help="output dir")
    parser.add_argument('--file_suffix', type=str, default='.jpg', help="suffix of rgb images")
    parser.add_argument('--window_size', type=int, default=3, help="window size for temporal smoothness")
    parser.add_argument('--rgb_threshold', type=int, default=20, help="consistency check, avoid misalignment")
    args = parser.parse_args()

    demo(args)



    