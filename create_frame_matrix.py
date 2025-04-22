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


import argparse
import glob, cv2, os
import numpy as np
from pypfm import PFMLoader
from scipy import signal


def forward_projection(points, image, extrinsic, intrinsic, h, w, mask=None):

    # z_buffer based forward projection
    trans = intrinsic.dot(np.linalg.inv(extrinsic))
    uvd = trans.dot(points)
    depth = uvd[2, :]
    depth = np.maximum(depth, 1e-10)

    u = np.uint16(np.round(uvd[0, :]/depth))
    v = np.uint16(np.round(uvd[1, :]/depth))
    image = image.reshape(-1 , 3)

    sort_idx = np.argsort(-depth)
    depth = depth[sort_idx]
    image = image[sort_idx, :]
    u = u[sort_idx]
    v = v[sort_idx]

    if mask is not None:
        mask = mask.reshape(-1, 1)
        mask = mask[sort_idx, :]

    # valid positions
    valid_d = np.where(depth > 1e-10)[0]
    valid_u = np.where((u<w) & (u>=0))[0]
    valid_v = np.where((v<h) & (v>=0))[0]
    valid_position = np.intersect1d(valid_d, valid_u)
    valid_position = np.intersect1d(valid_position, valid_v)

    u_valid = u[valid_position]
    v_valid = v[valid_position]
    image_valid = image[valid_position, :]
    depth_valid = depth[valid_position]

    # new view
    img_new = np.zeros((h, w, 3))
    img_new[v_valid, u_valid, :] = image_valid

    # depth
    depth_new = np.zeros((h,w))
    depth_new[v_valid, u_valid] = depth_valid

    # mask
    img_new_mask = np.zeros_like(depth_new)
    img_new_mask[v_valid, u_valid] = 1

    return img_new, depth_new[:,:,None], img_new_mask


def backward_projection(depth, uv, extrinsic, intrinsic, h, w):

    u = uv[0].flatten()
    v = uv[1].flatten()
    depth = depth.flatten()

    uvd = np.ones((4, int(h*w)))
    uvd[0, :] = u*depth
    uvd[1, :] = v*depth
    uvd[2, :] = depth
    trans = (np.linalg.inv(intrinsic)).dot(extrinsic)
    points = trans.dot(uvd)

    return points


def remove_cracks_isolated_points(image, mask, depth, kernel_size=3, threshold_isolated=0.5, threshold_crack=0.2, kernel_type='gaussian', num_planes=4):
    # Create multi-layer projection. default, 4 planes
    assert num_planes > 1
    if len(image.shape) > len(mask.shape):
        mask = mask[..., None]
    
    depth_median = np.median(depth[np.where(mask>0)])
    depth_max = np.max(depth[np.where(mask>0)])
    depth_min = np.min(depth[np.where(mask>0)])
    
    # divide into two groups
    num_planes_1 = int(num_planes/2)
    depth_interval_1 = (depth_median - depth_min)/num_planes_1
    num_planes_2 = num_planes - num_planes_1
    depth_interval_2 = (depth_max - depth_median)/num_planes_2
    
    split_at_depths = []
    for plane_idx in range(num_planes_1):
        split_at_depth = depth_min + (plane_idx + 1)*depth_interval_1
        split_at_depths.append(split_at_depth)
    
    for plane_idx in range(num_planes_2):
        split_at_depth = depth_median + (plane_idx + 1)*depth_interval_2
        split_at_depths.append(split_at_depth)

    layers = []
    layer_masks = []
    for i in range(len(split_at_depths)):
        if i == 0:
            layer_v, layer_u, _ = np.where(depth<=split_at_depths[i])
        else:
            layer_v, layer_u, _ = np.where((depth>split_at_depths[i-1]) & (depth<=split_at_depths[i]))
        layer_mask = np.zeros_like(mask)
        layer_mask[layer_v, layer_u, :] = 1
        layer_mask = mask*layer_mask
        layer = image*layer_mask
        layers.append(layer)
        layer_masks.append(layer_mask)
    
    def gaussian_kernel(kernel_size, std=1, normalised=True):
        gaussian1D = signal.gaussian(kernel_size, std)
        gaussian2D = np.outer(gaussian1D, gaussian1D)
        if normalised:
            gaussian2D /= (2*np.pi*(std**2))
        return gaussian2D


    def remove_isolated_points(image, mask, kernel_size=3, threshold=0.5):
        kernel_mean = np.ones((kernel_size,kernel_size), np.float32) / (kernel_size**2)
        frame_weight = mask[:,:,0] * cv2.filter2D(mask, -1, kernel_mean)
        mask_new = np.zeros_like(mask)
        mask_new[np.where(frame_weight > threshold)] = 1
        if len(image.shape) > len(mask_new.shape):
            mask_new = mask_new[..., None]
        image_new = image * mask_new
        return image_new, mask_new


    def fill_crack(image, mask, kernel_size, kernel_type='gaussian', threshold=0.2):
        if kernel_type == 'gaussian':
            kernel = gaussian_kernel(kernel_size=kernel_size)
        else:
            kernel = np.ones((kernel_size,kernel_size), np.float32) / (kernel_size**2)  # mean
            
        image_weight = cv2.filter2D(mask, -1, kernel)[..., None]
        image_blur = cv2.filter2D(image, -1, kernel)
        image_blur = image_blur / (image_weight+1e-6)

        mask_new = np.zeros_like(mask)
        mask_new[np.where(image_weight>threshold)] = 1.0
        mask = mask_new * mask
        image_new = image + 0
        image_new = image_new*mask + image_blur*(mask_new - mask)

        return image_new, mask_new

    # handle artifacts
    layers_new = []
    layer_masks_new = []
    for i in range(len(layers)):
        ## remove isolated points
        layer_new, layer_mask_new = remove_isolated_points(layers[i], layer_masks[i], kernel_size=kernel_size, threshold=threshold_isolated)
        
        ## remove cracks
        layer_new, layer_mask_new = fill_crack(layer_new, layer_mask_new, kernel_size=kernel_size, kernel_type=kernel_type, threshold=threshold_crack)

        layers_new.append(layer_new)
        layer_masks_new.append(layer_mask_new)
        
    # render multiplane images into the final image 
    ## begin with the last layer
    image_new = layers_new[-1] + 0
    mask_new = layer_masks_new[-1] + 0
    for i in range(len(layers_new)-1):
        i = len(layers_new) - 2 - i ## far to near
        mask_update_position = np.zeros_like(mask_new)
        mask_update_position[np.where(layer_masks_new[i]>0)] = 1
        mask_new = (1 - mask_update_position)*mask_new + mask_update_position*layer_masks_new[i]
        image_new = (1 - mask_update_position)*image_new + mask_update_position*layers_new[i]

    return image_new, mask_new

# Define cameras for creating frame matrix
def init_camera(args, num_frames):
    ## define the camera intrinsics
    intrinsic_matrix = np.array([
        [args.focal, 0, 0.5*args.width, 0],
        [0, args.focal, 0.5*args.height, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        ])
    intrinsic_matrix = intrinsic_matrix[None, ...]
    intrinsic_matrixs = intrinsic_matrix.repeat(num_frames, axis=0) # share the same intrinsic

    ## define the camera poses
    ## currently, support parallel cameras. Toe-in style camera is not implemented.
    extrinsic_matrixs = []
    if args.use_circle_trajectory:
        max_baseline = args.circle_diameter
        num_training_views = args.circle_training_views
        radius = max_baseline/2
        x_part1 = np.linspace(0, max_baseline, int(num_training_views/2 + 1))
        x_part2 = x_part1[1:-1][::-1]
        y_part1 = np.sqrt(radius**2 - (x_part1-radius)**2)
        y_part2 = -1 * np.sqrt(radius**2 - (x_part2-radius)**2)
        x_baselines = np.concatenate((x_part1, x_part2))
        y_baselines = np.concatenate((y_part1, y_part2)) * args.circle_scale_y
        
        for j in range(num_training_views):
            x_baseline = x_baselines[j]
            y_baseline = y_baselines[j]
            # camera2world
            extrinsic_t = np.array([[1, 0, 0, x_baseline], 
                                    [0, 1, 0, y_baseline],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
            extrinsic_matrixs.append(extrinsic_t)
    else:
        
        if args.middle2leftright:
            baselines = np.linspace(-0.5*args.max_baseline, 0.5*args.max_baseline, args.num_training_views)
            baselines[int(args.num_training_views/2)] = 0
        elif args.left2right:
            baselines = np.linspace(0, args.max_baseline, args.num_training_views)
        elif args.right2left:
            baselines = np.linspace(0, -args.max_baseline, args.num_training_views)
        else:
            baselines = np.linspace(0, -args.max_baseline, args.num_training_views)
        
        for j in range(args.num_training_views):
            baseline = baselines[j]
            # camera2world
            extrinsic_t = np.array([[1, 0, 0, baseline], 
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
            extrinsic_matrixs.append(extrinsic_t)
            
    return intrinsic_matrixs, extrinsic_matrixs


def main(args):
    
    # load the imgs and estimated depths
    frames = []
    depths = []
    frame_names = sorted(glob.glob(args.img_path + '/images/*.jpg') + glob.glob(args.img_path + '/images/*.png'))
    if args.use_refined_depth:
        if args.depth_suffix == '.npy':
            depth_names = sorted(glob.glob(args.depth_path + '/depths_refined/*.npy'))
        elif args.depth_suffix == '.png':
            depth_names = sorted(glob.glob(args.depth_path + '/depths_refined/*.png'))
        else:
            depth_names = sorted(glob.glob(args.depth_path + '/depths_refined/*.pfm')) 
    else:
        if args.depth_suffix == '.npy':
            depth_names = sorted(glob.glob(args.depth_path + '/depths/*.npy'))
        elif args.depth_suffix == '.png':
            depth_names = sorted(glob.glob(args.depth_path + '/depths/*.png'))
        else:
            depth_names = sorted(glob.glob(args.depth_path + '/depths/*.pfm')) 

    for i in range(args.num_frames):
        i = i + 0  # start idx
        
        frame_name = frame_names[i]
        img = cv2.imread(frame_name)[:,:,::-1]/255.0  # to RGB
        
        depth_name = depth_names[i]  # disparity
        if depth_name.endswith('png'):
            depth = cv2.imread(depth_name, -1)[..., None]  # relative depth
        elif depth_name.endswith('pfm'):
            loader = PFMLoader(color=False, compress=False)
            depth = loader.load_pfm(depth_name)
            depth = np.flipud(depth)[..., None]
        else:
            depth = np.load(depth_name)[..., None]
        depth = np.maximum(depth, 1e-6)  # require > 0
        frames.append(img)
        depths.append(depth)
    
    frames = np.stack(frames, axis=0)  # NxHxWxC, bgr
    depths = np.stack(depths, axis=0) # NxHxWx1
    num_frames, frame_h, frame_w, _ = frames.shape

    ## normalize the depth value
    if args.use_metric_depth:
        depths_normlized = depths
    else:
        depths_normlized = (depths - depths.min())/(depths.max() - depths.min())  # normalize to 0~1
        if args.nearest_depth > 0:
            depths_normlized = depths_normlized / args.nearest_depth
        if args.farest_depth > 0:
            depths_normlized = depths_normlized + (1/args.farest_depth)
    depths = depths_normlized

    
    # define cameras for frame matrix
    intrinsics, extrinsics = init_camera(args, num_frames)
    
    # create output folder
    max_baseline = args.circle_diameter if args.use_circle_trajectory else args.max_baseline
    # if args.use_refined_depth:
    #     output_path = args.output_path + '/refined_train_views_fov%s_baseline%s_far%s/' % (intrinsics[0, 0, 0], max_baseline, args.farest_depth)
    # else:
    #     output_path = args.output_path + '/train_views_fov%s_baseline%s_far%s/' % (intrinsics[0, 0, 0], max_baseline, args.farest_depth)
    # os.makedirs(output_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    
    # warp reference view to the target view
    ## source view, camera-to-world
    extrinsic_s = np.array([[1, 0, 0, 0], 
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    
    intrinsic_s = intrinsics[0]
    intrinsic_t = intrinsic_s  # share
    
    for j in range(args.num_training_views):
        extrinsic_t = extrinsics[j]
        for i in range(num_frames):
            img = frames[i, :, :, :]
            depth = depths[i, :, :, :]  # disparity
            depth = 1.0/(depth+1e-6)
            img_h, img_w, _ = img.shape
            u = np.linspace(0, img_w-1, img_w)
            v = np.linspace(0, img_h-1, img_h)
            uv = np.meshgrid(u, v)
            points = backward_projection(depth, uv, extrinsic_s, intrinsic_s, img_h, img_w)
            img_t, depth_t, img_t_mask = forward_projection(points, img, extrinsic_t, intrinsic_t, img_h, img_w)

            ## handle artifacts
            img_t, img_t_mask = remove_cracks_isolated_points(img_t, img_t_mask, depth_t, num_planes=args.num_planes)
            
            if (extrinsic_t == extrinsic_s).all():
                img_t = img
                img_t_mask[:,:] = 1

            cv2.imwrite(args.output_path + '/%05d_baseline_%s_%03d.jpg' % (i, max_baseline, j), np.uint8(img_t[:,:,::-1]*255.0))
            cv2.imwrite(args.output_path + '/%05d_baseline_%s_%03d_mask.png' % (i, max_baseline, j), np.uint8(img_t_mask*255.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', type=int, default=16)  # 24
    parser.add_argument('--img_path', type=str, default='./Videos/Obama_is_speaking_320x576/') 
    parser.add_argument('--depth_path', type=str, default='./Videos/Obama_is_speaking_320x576/')
    parser.add_argument('--output_path', type=str, default='./Videos/frame_matrix/Obama_is_speaking/')
    parser.add_argument('--use_refined_depth', type=int, default=1)  # 1: use smoothed depth; 0: use your own or predicted depth
    parser.add_argument('--depth_suffix', type=str, default='.npy')
    parser.add_argument('--width', type=int, default=576)  # 512
    parser.add_argument('--height', type=int, default=320) # 512
    parser.add_argument('--focal', type=float, default=800) # 400, 600, 800
    parser.add_argument('--max_baseline', type=float, default=0.08) # 0.05, 0.06, 0.07, 0.08
    parser.add_argument('--num_training_views', type=int, default=8)  # number of camera views (frame matrix). 4, 6, 8
    parser.add_argument('--farest_depth', type=float, default=10) 
    parser.add_argument('--nearest_depth', type=float, default=-1) # -1: nearest depth is ~1m. can use 2m, 3m, ....
    parser.add_argument('--use_metric_depth', action='store_true', help='True: use metric depth')  # provided as 1/depth
    parser.add_argument('--middle2leftright', action='store_true', help='True: reference video as the middle view') 
    parser.add_argument('--left2right', action='store_true', help='True: reference video is the left-view video') 
    parser.add_argument('--right2left', action='store_true', default=False, help='True: reference video is the right-view video') 
    parser.add_argument('--use_circle_trajectory', action='store_true')
    parser.add_argument('--circle_diameter', type=float, default=0.075)
    parser.add_argument('--circle_training_view', type=int, default=16)
    parser.add_argument('--circle_scale_y', type=float, default=0.5)
    parser.add_argument('--num_planes', type=int, default=4)  # project points onto xxx planes for artifacts removal

    args = parser.parse_args()
    main(args)
    