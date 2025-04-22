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
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, DDPMScheduler, StableVideoDiffusionPipeline
import torch, cv2

from transformers import CLIPTokenizer, CLIPTextModel
import glob, os, math, random

class SVGDiffusion:
    def __init__(self):
        self.parse_args()
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, required=True, help="The target text prompt")
        parser.add_argument("--num_frames", type=int, default=16, help="The number of frames")
        parser.add_argument("--init_image", type=str, default='./Videos/frame_matrix/xxx', help="The path to the input frame matrix")
        parser.add_argument("--output_root", type=str, default='./results/',help="The path to save results")
        parser.add_argument("--model_path", type=str, default="cerspense/zeroscope_v2_576w", help="Video generation model. The path to the HuggingFace model",)
        parser.add_argument("--batch_size", type=int, default=1, help="The number of videos to generate")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--low_ram_vae", type=int, default=0, help="save ram for video diffusion",)
        parser.add_argument("--num_inference_steps", type=int, default=50, help="The number of denoising inference steps")
        parser.add_argument("--downscale", type=int, default=8, help="downsample scale from image to latent, zeroscope is 8")
        parser.add_argument("--threshold", type=float, default=0.7, help="detremine known and unknown regions")
        parser.add_argument("--resample", type=int, default=6, help="resample xxx times")
        parser.add_argument("--num_views", type=int, default=8, help="number of camera views, match the frame matrix creation")
        parser.add_argument('--negative_prompt', type=str, default='low-quality, blurry, flickering, ghost', help='') 
        parser.add_argument('--soft_mask_k', type=int, default=5, help='kernel size used to create soft mask') 
        parser.add_argument('--soft_mask_type', type=str, default='gaussian', help='kernel type') 
        parser.add_argument('--frame_matrix_end', type=int, default=600, help='stop denoising frame matrix at xxx') 
        parser.add_argument('--update_source_latent', type=bool, default=True, help='True: apply disocclusion boundary re-injection') 
        parser.add_argument('--update_source_latent_end', type=int, default=0, help='stop disocclusion boundary re-injection at xxx.') 
        parser.add_argument('--last_view_idx', type=int, default=7, help='another view idx to form stereoscopic video [0, another view idx], e.g., num_views - 1') 
        parser.add_argument('--monocular_video_idx', type=int, default=0, help='0 if middle2leftright is False, else int(num_views/2)') 
        parser.add_argument('--last_view_resample', type=int, default=2, help='resample xxx times for the last view') 
        parser.add_argument('--erode_size', type=int, default=1, help='erode the warped image a little considering the inaccurate depth')
        parser.add_argument('--middle2leftright', action='store_true', help='reference video is projected into left and right views')
        parser.add_argument('--right2left', action='store_true', help='reference video is the right-view video')  
        parser.add_argument('--height', type=int, default=320, help='image height') 
        parser.add_argument('--width', type=int, default=576, help='image width') 
        parser.add_argument('--latent_h', type=int, default=40, help='latent height, image height/downscale') 
        parser.add_argument('--latent_w', type=int, default=72, help='latent width') 
        
        # for multi-view video, frame_matrix_end <= 0
        parser.add_argument('--fix_last_view_begin', type=int, default=600, help='exclusively denosie the last view after xxx') 
        parser.add_argument('--stage2_resample', type=int, default=4, help='resample xxx times')
        parser.add_argument('--stage2_time_direction_denoise_freq', type=int, default=3, help='denoise in time direction every xxx resamples')
        
        self.args = parser.parse_args()
        
    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(self.args.model_path, torch_dtype=torch.float16)
        # pipe = DiffusionPipeline.from_pretrained(self.args.model_path, torch_dtype=torch.float16)
        self.vae = pipe.vae.to(self.args.device)
        self.unet = pipe.unet.to(self.args.device)

        # use stable video diffusion's decoder (deflickering)
        ## has similar results. use zeroscope's decoder, comment the following three lines
        pipe1 = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe1.enable_model_cpu_offload()
        self.vae = pipe1.vae.to(self.args.device)

        # tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.model_path, subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.model_path, subfolder="text_encoder", torch_dtype=torch.float16)
        self.text_encoder = self.text_encoder.to(self.args.device)

        # DDIM scheduler
        # self.scheduler = DDIMScheduler.from_pretrained(
        #     self.args.model_path,
        #     subfolder="scheduler",
        #     torch_dtype=torch.float16,)

        
        # DDPM scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            self.args.model_path,
            subfolder="scheduler",
            torch_dtype=torch.float16,
            )

        self.alphas = self.scheduler.alphas_cumprod.to(self.args.device)

    def load_video_images(self, frame_path, h=320 ,w=576, latent_h=40, latent_w=72, num_frames=16, suffix='.jpg', downscale=8, threshold=0.5, num_views=8, soft_mask_k=3, soft_mask_type='gaussian', erode_size=3):
        frames_all = []
        ori_masks_all = []
        latent_masks_all = []

        def create_latent_mask(masks_ori, latent_h, latent_w, down_scale=8, threshold=0.5, enable_remove=True):
            """
            create mask in latent space
            """
            latent_masks = []
            
            # apply conv to roughly determine regions affected by unknown regions.
            kernel = np.ones((down_scale, down_scale), np.float32) / (down_scale**2)
            for i in range(masks_ori.shape[0]):
                mask_cur = masks_ori[i]
                if enable_remove and down_scale > 1:
                    mask_cur = cv2.filter2D(masks_ori[i], -1, kernel)
                    # remove unreliable positions, controled by threshold.
                    ## reference video in the frame matrix retains all information. 
                    mask_cur[np.where(mask_cur<threshold)] = 0  
                    mask_cur[np.where(mask_cur>=threshold)] = 1  
                    mask_cur = mask_cur*masks_ori[i,:,:,0]
                latent_mask = cv2.resize(mask_cur, (latent_w, latent_h))
                latent_mask[np.where(latent_mask < 1)] = 0
                latent_masks.append(latent_mask)
            latent_masks = np.stack(latent_masks)
            if len(latent_masks.shape) == 4:
                return latent_masks[:,:,:,0:1]
            else:
                return latent_masks[..., None]

        def create_soft_mask(mask, k_size, soft_mask_type='gaussian'):
            '''
            soft mask smoothly combines the known and unknown regions
            '''
            soft_masks = []
            for i in range(mask.shape[0]):
                if soft_mask_type == 'gaussian':
                    soft_mask = cv2.GaussianBlur(mask[i], (k_size,k_size), 0)
                else:
                    soft_mask = cv2.blur(mask[i], (k_size, k_size))
                soft_masks.append(soft_mask)
            soft_masks = np.stack(soft_masks)
            soft_masks = soft_masks[..., None] 
            soft_masks = soft_masks * mask
            return soft_masks

        for view_idx in range(num_views):
            frame_names = sorted(glob.glob(frame_path + '/*%03d%s' % (view_idx, suffix)))
            frames = []
            ori_masks = []
            latent_masks = []
            kernel = np.ones((erode_size, erode_size), np.float32) / erode_size*erode_size

            for i in range(num_frames):
                # load rgb images, 0~1
                frame_name = frame_names[i]
                img = cv2.imread(frame_name)[:,:,::-1]/255.0   # RGB with range: 0~1
                img = cv2.resize(img, (w, h))
                frames.append(img)
                
                # load mask images, 0 or 1, 0 is unknown region.
                # erode a liitle bit, when boundary is not accurate
                mask = cv2.imread(frame_name.replace(suffix, '_mask.png'), -1)/255.0
                ori_mask = cv2.resize(mask, (w, h))
                if erode_size > 1:
                    ori_mask = cv2.erode(ori_mask, kernel)  
                ori_mask[np.where(ori_mask < 1)] = 0
                ori_masks.append(ori_mask)

            frames = np.stack(frames)
            ori_masks = np.stack(ori_masks)
            if len(frames.shape) != len(ori_masks.shape):
                ori_masks = ori_masks[..., None]
            frames = frames*ori_masks

            # resize the mask, match the size of latent feature
            latent_masks = create_latent_mask(ori_masks, latent_h, latent_w, down_scale=downscale, threshold=threshold)
            if soft_mask_k >= 1:
                latent_masks = create_soft_mask(latent_masks, soft_mask_k, soft_mask_type)          
            frames_all.append(frames)
            ori_masks_all.append(ori_masks)
            latent_masks_all.append(latent_masks)

        frames_all = np.stack(frames_all)
        ori_masks_all = np.stack(ori_masks_all)
        latent_masks_all = np.stack(latent_masks_all)

        return frames_all, ori_masks_all, latent_masks_all

    @torch.no_grad()
    def denoising_process(
        self,
        image_path,
        prompts,
        num_frames = 24,
        height=320,
        width=576,
        latent_h=40,
        latent_w=72,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        threshold=0.5,
        downscale=8,
        resample=8,
        num_views=8,
        negative_prompt="",
        soft_mask_k=3,
        soft_mask_type='gaussian',
        frame_matrix_end=500,
        update_source_latent=True,
        update_source_latent_end=0,
        fix_last_view_begin=500,
        last_view_resample=4,
        stage2_resample=8,
        stage2_time_direction_denoise_freq=3,
        erode_size=3,
        last_view_idx=None,
        middle2leftright=False,
        monocular_video_idx=0,
        right2left=False,
    ):
        

        def predict_noise(latents, t, text_embeddings, guidance_scale):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # # visualize the intermidiate latent feature
            # visualize_latent(t, latents, noise_pred, output_path='./results/visualization_latent/')

            return noise_pred

        def disocclusion_boundary_reinjection(pred_original_latents, source_latents, latent_masks, images, ori_masks):
            # decode clean latents into image space
            pred_original_latents = 1 / self.vae.config.scaling_factor * pred_original_latents
            batch_size, channels, num_frames, height, width = pred_original_latents.shape
            pred_original_latents = pred_original_latents.permute(0, 2, 1, 3, 4).reshape(num_frames*batch_size, channels, height, width)
            pred_original_images = self.vae.decode(pred_original_latents, num_frames).sample
            pred_original_images = (pred_original_images / 2 + 0.5).permute(0, 2, 3, 1).clamp(0, 1)

            # replace in image space
            pred_original_images = pred_original_images * (1-ori_masks) + images * ori_masks

            # update known features by encoding again
            with torch.no_grad():
                source_latents_updated = self.images2latents(pred_original_images, use_mean=False)
            return source_latents_updated
        
        def ddpm_resample(latents_cur, t_next, t_cur, resample, source_latents, latent_masks, text_embeddings, guidance_scale, num_frames=16, num_views=8, last_view_idx=7, last_view_resample=4, fix_last_view_begin=-1, frame_matrix_end=500, stage2_resample=8):

            if resample==0:
                return latents_cur
            
            if t_next < frame_matrix_end:
                resample = last_view_resample
            
            alpha_t_cur = self.scheduler.alphas_cumprod[t_cur]
            alpha_t_next = self.scheduler.alphas_cumprod[t_next]
            noisy_source_latents = self.scheduler.add_noise(source_latents, torch.randn_like(source_latents), t_next)  

            # stage 2 denoising, for multi-view video generation
            if t_next <= fix_last_view_begin and (frame_matrix_end <= 0):
                stage2_denoise = True
                resample = stage2_resample
                for i in range(last_view_resample):
                    ## Markov scheme, add nosie step by step, back to t_next. (z_{t-1} -> z_t)
                    betas = self.scheduler.betas
                    alpha_cumprod = 1
                    for j in range(int(t_next-t_cur)):
                        beta = betas[t_cur+j]
                        alpha = 1-beta
                        alpha_cumprod = alpha*alpha_cumprod
                    
                    noise = torch.randn_like(latents_cur)
                    latents = alpha_cumprod.sqrt()*latents_cur + (1-alpha_cumprod).sqrt()*noise
 
                    for view_idx in range(num_views):
                        latents[view_idx] = latents[view_idx] * (1 - latent_masks[view_idx])  + \
                        self.scheduler.add_noise(source_latents[view_idx], torch.randn_like(latents[0]), t_next) * latent_masks[view_idx]

                    # Denoise the last view exclusively
                    t = t_next
                    print('timestep next: %s' % t)

                    denoise_accmulated = torch.zeros_like(latents)
                    operation_accmulated = torch.zeros_like(latents)

                    for view_idx in [last_view_idx]:
                        noise_pred = predict_noise(latents[view_idx, :, :, :, :, :], t, text_embeddings, guidance_scale)
                        denoise_accmulated[view_idx, :, :, :, :, :] = denoise_accmulated[view_idx, :, :, :, :, :] + self.scheduler.step(noise_pred, t, latents[view_idx, :, :, :, :, :]).prev_sample
                        operation_accmulated[view_idx, :, :, :, :, :] =  operation_accmulated[view_idx, :, :, :, :, :] + 1

                    operation_accmulated[torch.where(operation_accmulated==0)] = 1
                    latents_cur[last_view_idx, :, :, :, :, :] = (denoise_accmulated / operation_accmulated)[last_view_idx, :, :, :, :, :]  # update the last view
            else:
                stage2_denoise = False 

            # begin resample
            for i in range(int(resample)):
                ## Markov scheme, add nosie step by step, back to t_next. (z_(t-1) -> z_t)
                betas = self.scheduler.betas
                alpha_cumprod = 1
                for j in range(int(t_next-t_cur)):
                    beta = betas[t_cur+j]
                    alpha = 1-beta
                    alpha_cumprod = alpha*alpha_cumprod
                
                noise = torch.randn_like(latents_cur)
                latents = alpha_cumprod.sqrt()*latents_cur + (1-alpha_cumprod).sqrt()*noise
                
                ## combine unknown and known regions
                for view_idx in range(num_views):
                    latents[view_idx] = latents[view_idx] * (1 - latent_masks[view_idx])  + \
                    self.scheduler.add_noise(source_latents[view_idx], torch.randn_like(latents[0]), t_next) * latent_masks[view_idx]
                
                ## denoise again
                t = t_next
                print('timestep %s' % t)
        
                denoise_accmulated = torch.zeros_like(latents)
                operation_accmulated = torch.zeros_like(latents)
                denoise_accmulated_last_view = None

                if t_next > frame_matrix_end:
                    # denoise in different directions
                    if not stage2_denoise:
                        if i%4 == 0 or i%4 == 2:
                            # denoise in time direction
                            for view_idx in range(num_views):
                                noise_pred = predict_noise(latents[view_idx, :, :, :, :, :], t, text_embeddings, guidance_scale)
                                denoise_accmulated[view_idx, :, :, :, :, :] = denoise_accmulated[view_idx, :, :, :, :, :] + self.scheduler.step(noise_pred, t, latents[view_idx, :, :, :, :, :]).prev_sample
                                operation_accmulated[view_idx, :, :, :, :, :] =  operation_accmulated[view_idx, :, :, :, :, :] + 1

                        elif i%4 == 1:                  
                            # denoise in spatial direction, view_0 to view_N
                            for time_idx in range(num_frames):
                                noise_pred = predict_noise(latents[:, :, :, time_idx, :, :].permute(1,2,0,3,4), t, text_embeddings, guidance_scale)
                                denoise_accmulated[:, :, :, time_idx, :, :] = denoise_accmulated[:, :, :, time_idx, :, :] + (self.scheduler.step(noise_pred, t, latents[:, :, :, time_idx, :, :].permute(1,2,0,3,4)).prev_sample).permute(2,0,1,3,4)
                                operation_accmulated[:, :, :, time_idx, :, :] =  operation_accmulated[:, :, :, time_idx, :, :] + 1
                            
                        elif i%4 == 3:
                            # denoise in spatial direction, view_N to view_0
                            for time_idx in range(num_frames):
                                noise_pred = torch.flip(predict_noise(torch.flip(latents[:, :, :, time_idx, :, :], dims=[0]).permute(1,2,0,3,4), t, text_embeddings, guidance_scale), dims=[2])
                                denoise_accmulated[:, :, :, time_idx, :, :] = denoise_accmulated[:, :, :, time_idx, :, :] + (self.scheduler.step(noise_pred, t, latents[:, :, :, time_idx, :, :].permute(1,2,0,3,4)).prev_sample).permute(2,0,1,3,4)
                                operation_accmulated[:, :, :, time_idx, :, :] =  operation_accmulated[:, :, :, time_idx, :, :] + 1
                    else:
                        # stage2 denoising
                        ## change the frequency of spatial and time directions
                        if (i+1) % stage2_time_direction_denoise_freq == 0: 
                            print('stage2: update time direction')
                            for view_idx in range(num_views):
                                noise_pred = predict_noise(latents[view_idx, :, :, :, :, :], t, text_embeddings, guidance_scale)
                                denoise_accmulated[view_idx, :, :, :, :, :] = denoise_accmulated[view_idx, :, :, :, :, :] + self.scheduler.step(noise_pred, t, latents[view_idx, :, :, :, :, :]).prev_sample
                                operation_accmulated[view_idx, :, :, :, :, :] =  operation_accmulated[view_idx, :, :, :, :, :] + 1
                        else:
                            print("stage2: update spatial direction")
                            if i % 2 == 0:
                                # update the spatial direction, view_0 to view_N
                                for time_idx in range(num_frames):
                                    noise_pred = predict_noise(latents[:, :, :, time_idx, :, :].permute(1,2,0,3,4), t, text_embeddings, guidance_scale)
                                    denoise_accmulated[:, :, :, time_idx, :, :] = denoise_accmulated[:, :, :, time_idx, :, :] + (self.scheduler.step(noise_pred, t, latents[:, :, :, time_idx, :, :].permute(1,2,0,3,4)).prev_sample).permute(2,0,1,3,4)
                                    operation_accmulated[:, :, :, time_idx, :, :] =  operation_accmulated[:, :, :, time_idx, :, :] + 1
                            
                            if i % 2 == 1:
                                # update the spatial direction, view_N to view_0
                                for time_idx in range(num_frames):
                                    noise_pred = torch.flip(predict_noise(torch.flip(latents[:, :, :, time_idx, :, :], dims=[0]).permute(1,2,0,3,4), t, text_embeddings, guidance_scale), dims=[2])
                                    denoise_accmulated[:, :, :, time_idx, :, :] = denoise_accmulated[:, :, :, time_idx, :, :] + (self.scheduler.step(noise_pred, t, latents[:, :, :, time_idx, :, :].permute(1,2,0,3,4)).prev_sample).permute(2,0,1,3,4)
                                    operation_accmulated[:, :, :, time_idx, :, :] =  operation_accmulated[:, :, :, time_idx, :, :] + 1
                        
                        ## fix last view
                        operation_accmulated[last_view_idx, :, :, :, :, :] = 1
                        denoise_accmulated[last_view_idx, :, :, :, :, :] = latents_cur[last_view_idx, :, :, :, :, :]        
                else:
                    # only denoise required views
                    if middle2leftright:
                        view_idxs = [0, last_view_idx]
                    else:
                        view_idxs = [last_view_idx]
                        
                    for view_idx in view_idxs:
                        noise_pred = predict_noise(latents[view_idx, :, :, :, :, :], t, text_embeddings, guidance_scale)
                        denoise_accmulated[view_idx, :, :, :, :, :] = denoise_accmulated[view_idx, :, :, :, :, :] + self.scheduler.step(noise_pred, t, latents[view_idx, :, :, :, :, :]).prev_sample
                        operation_accmulated[view_idx, :, :, :, :, :] =  operation_accmulated[view_idx, :, :, :, :, :] + 1

                operation_accmulated[torch.where(operation_accmulated==0)] = 1
                latents_cur = denoise_accmulated/operation_accmulated  # finish one resample

            return latents_cur

        def save_results(latents_all, save_idxs, is_list=True, step_idx=None):
            for view_idx in save_idxs:
                if is_list:
                    latents = latents_all[view_idx]
                else:
                    latents = latents_all
                batch_size, channels, num_frames, height, width = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(num_frames*batch_size, channels, height, width)
                results = self.vae.decode(latents, num_frames).sample

                results = (results / 2 + 0.5).clamp(0, 1)
                results = results.detach().cpu().permute(0, 2, 3, 1).numpy()
                results = (results * 255).round().astype("uint8")

                save_frames(results, path=self.args.output_root, view_idx=view_idx, step_idx=step_idx)
            return results
        

        batch_size = len(prompts)
        if last_view_idx is None:
            last_view_idx = num_views - 1

        # load video and convert to latent features
        images_all, ori_masks_all, latent_masks_all = self.load_video_images(image_path, h=height, w=width, latent_h=latent_h, latent_w=latent_w, num_frames=num_frames, downscale=downscale, threshold=threshold, num_views=num_views, soft_mask_k=soft_mask_k, soft_mask_type=soft_mask_type, erode_size=erode_size)                                                         
        num_views, num_frames, _, _, _ = images_all.shape
        images_all = torch.from_numpy(images_all).to('cuda')
        source_latents_all = []   # source_latents_all contains known features
        for view_idx in range(num_views):
            with torch.no_grad():
                source_latents_all.append(self.images2latents(images_all[view_idx], use_mean=False))
        source_latents_all = torch.stack(source_latents_all)

        # reshape mask to match source latent
        ori_masks_all = torch.from_numpy(ori_masks_all).to('cuda').half()
        latent_masks_all = torch.from_numpy(latent_masks_all).to('cuda').half().unsqueeze(0).permute(1,0,5,2,3,4) # num_view X bacth X channel X num_frames X height X width

        # tokenizer and text embeddings
        ## text condition
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]  # Bx77x1024

        ## no text condition
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [negative_prompt] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]

        ## concate conditon and uncondition
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # set denosing step
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps[int(len(self.scheduler.timesteps) * 0) :]

        # create random noise for generation
        ## generate unknow part from random noise
        latents_all = torch.randn(source_latents_all.shape, generator=generator,) 
        latents_all = latents_all.to("cuda").half()

        # begin the denoising process
        for i in range(len(timesteps)):
            t = timesteps[i]
            if (i+1) < len(timesteps):
                t_prev = timesteps[i+1]
            else:
                t_prev = None
            print('timestep %s' % t)

            denoise_accmulated = torch.zeros_like(latents_all)
            operation_accmulated = torch.zeros_like(latents_all) # counter
            pred_original_latents_all = source_latents_all + 0  # clean latent

            # begin denoising
            if t > frame_matrix_end:
                view_idxs = range(num_views)  # camera views for denoising
            else:
                if middle2leftright:
                    view_idxs = [0, last_view_idx]  # denoise both left view and right view
                else:
                    view_idxs = [last_view_idx]  # denoise the last view (right view for left2right, left view for right2left)
            
            for view_idx in view_idxs:
                noise_pred = predict_noise(latents_all[view_idx, :, :, :, :, :], t, text_embeddings, guidance_scale)
                outputs_after_step = self.scheduler.step(noise_pred, t, latents_all[view_idx, :, :, :, :, :]) # z_t -> z_(t-1)
                denoise_accmulated[view_idx, :, :, :, :, :] = denoise_accmulated[view_idx, :, :, :, :, :] + outputs_after_step.prev_sample  # z_(t-1)
                pred_original_latents_all[view_idx, :, :, :, :, :] = outputs_after_step.pred_original_sample # z_0
                operation_accmulated[view_idx, :, :, :, :, :] =  operation_accmulated[view_idx, :, :, :, :, :] + 1

            operation_accmulated[torch.where(operation_accmulated==0)] = 1
            latents_all = denoise_accmulated/operation_accmulated

            # disocclusion boundary re-injection
            if update_source_latent and (t > update_source_latent_end):
                # update known features of frame matrix
                for view_idx in view_idxs:
                    source_latents_all[view_idx] = disocclusion_boundary_reinjection(pred_original_latents_all[view_idx], source_latents_all[view_idx], latent_masks_all[view_idx], images_all[view_idx], ori_masks_all[view_idx])

            # resmaple + denoising in time and spatial directions
            if t_prev is None:
                pass
            else:
                latents_all = ddpm_resample(latents_cur=latents_all, t_next=t, t_cur=t_prev, resample=resample, source_latents=source_latents_all, latent_masks=latent_masks_all, 
                                            text_embeddings=text_embeddings, guidance_scale=guidance_scale, num_frames=num_frames, num_views=num_views, 
                                            fix_last_view_begin=fix_last_view_begin, last_view_resample=last_view_resample, last_view_idx=last_view_idx,
                                            frame_matrix_end=frame_matrix_end, stage2_resample=stage2_resample)

            # add gaussian nosie to clean latents, combine known and unknown regions
            if t_prev is None:
                for view_idx in range(num_views):
                    latents_all[view_idx] = latents_all[view_idx] * (1 - latent_masks_all[view_idx]) + source_latents_all[view_idx] * latent_masks_all[view_idx]
            else:
                for view_idx in range(num_views):
                    latents_all[view_idx] = latents_all[view_idx] * (1 - latent_masks_all[view_idx]) + \
                    self.scheduler.add_noise(source_latents_all[view_idx], torch.randn_like(latents_all[0]), t_prev) * latent_masks_all[view_idx]
                    
            # save intermidiate results
            if i % 5 == 0:
                with torch.no_grad():
                    latents_tmp = pred_original_latents_all + 0
                    for view_idx in range(num_views):
                        latents_tmp[view_idx] = latents_tmp[view_idx] * (1 - latent_masks_all[view_idx]) + source_latents_all[view_idx] * latent_masks_all[view_idx]

                    save_results(latents_all=1 / self.vae.config.scaling_factor * latents_tmp, save_idxs=[last_view_idx], step_idx=t)
                           
        # save results
        with torch.no_grad():
            ## save frame matrix
            if frame_matrix_end > 0:
                save_idxs = [0, last_view_idx]
            else:
                save_idxs = range(num_views) 
            save_results(latents_all=1 / self.vae.config.scaling_factor * latents_all, save_idxs=save_idxs)

        # save stereoscopic video
        if right2left:
            save_video(frames_root=self.args.output_root, path=self.args.output_root, left_view_id=last_view_idx, right_view_id=0, num_frames=num_frames, w=width, h=height)
        else:
            save_video(frames_root=self.args.output_root, path=self.args.output_root, left_view_id=0, right_view_id=last_view_idx, num_frames=num_frames, w=width, h=height)
        
        return images_all.cpu().numpy()

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def images2latents(self, imgs, use_mean=False):
        imgs = imgs.float() * 2 -1  # normalize to -1~1
        imgs = imgs.unsqueeze(0) # BxNxHxWxC
        batch_size, num_frames, height, width, channels = imgs.shape

        # the shape feed into the video encoder should be "B*N C H W"
        imgs = imgs.permute(0, 1, 4, 2, 3).reshape(batch_size * num_frames, channels, height, width)
        imgs = imgs.half()
        input_dtype = imgs.dtype

        if self.args.low_ram_vae > 0:
            vnum = self.args.low_ram_vae
            mask_vae = torch.randperm(imgs.shape[0]) < vnum
            with torch.no_grad():
                if use_mean:
                   posterior_mask = torch.cat(
                        [
                            self.vae.encode(imgs[~mask_vae][i : i + 1].to(input_dtype)).latent_dist.mean 
                            for i in range(imgs.shape[0] - vnum)
                        ],
                        dim=0,
                    )
                else:
                    posterior_mask = torch.cat(
                        [
                            self.vae.encode(imgs[~mask_vae][i : i + 1].to(input_dtype)).latent_dist.sample()  
                            for i in range(imgs.shape[0] - vnum)
                        ],
                        dim=0,
                    )

            if use_mean:
                posterior = torch.cat(
                    [
                        self.vae.encode(imgs[mask_vae][i : i + 1].to(input_dtype)).latent_dist.mean 
                        for i in range(vnum)
                    ],
                    dim=0,
                )
            else:
                posterior = torch.cat(
                    [
                        self.vae.encode(imgs[mask_vae][i : i + 1].to(input_dtype)).latent_dist.sample()
                        for i in range(vnum)
                    ],
                    dim=0,
                )

            posterior_full = torch.zeros(
                imgs.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            posterior_full[~mask_vae] = posterior_mask
            posterior_full[mask_vae] = posterior
            latents = posterior_full * self.vae.config.scaling_factor
        else:
            if use_mean:
                posterior = self.vae.encode(imgs.to(input_dtype)).latent_dist
                latents = posterior.mean * self.vae.config.scaling_factor  # belnded latent diffusion
            else:
                posterior = self.vae.encode(imgs.to(input_dtype)).latent_dist
                latents = posterior.sample() * self.vae.config.scaling_factor  # 4dfy

        # the shape of latent from vae is b*Nx4x40x72

        latents = (latents[None, :].reshape((batch_size, num_frames, -1,) + latents.shape[2:]).permute(0, 2, 1, 3, 4))
        return latents.to(input_dtype)


def sliding_window(total_frames=24, frame_batch=16, overlapping=8):
    """
    handle long videos
    """
    return 0

def save_frames(frames, path, view_idx=0, step_idx=None):
    import os
    os.makedirs(path, exist_ok=True)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        frame = frames[i]
        frame = frame[:,:,::-1]
        if step_idx is None:
            cv2.imwrite(path + '/view%03d_%05d.jpg' % (view_idx, i), np.uint8(frame))
        else:
            cv2.imwrite(path + '/step%s_view%03d_%05d.jpg' % (step_idx, view_idx, i), np.uint8(frame))

def save_video(frames_root, path, left_view_id=0, right_view_id=7, num_frames=16, h=320, w=576, num_loop=20):
    import os, imageio
    os.makedirs(path, exist_ok=True)

    left_view_names = sorted(glob.glob(frames_root + '/view%03d_*.jpg' % left_view_id))
    right_view_names = sorted(glob.glob(frames_root + '/view%03d_*.jpg' % right_view_id))

    frames = []
    for i in range(num_frames):
        name_source = left_view_names[i]
        name = right_view_names[i]
        img_source = cv2.imread(name_source)
        img = cv2.imread(name)

        frame = np.concatenate([img_source, img], axis=1)
        frames.append(np.uint8(frame))

    # save mp4
    out = cv2.VideoWriter(path + "/results3d.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w*2, h))
    for i in range(num_loop):
        for frame in frames:
            out.write(np.uint8(frame))
    out.release()

    # save gif
    frames_gif = []
    for frame in frames:
        frames_gif.append(frame[:,:,::-1])
    imageio.mimsave(path + '/results3d.gif', frames_gif, fps=4, loop=1000)


if __name__ == "__main__":
    svg = SVGDiffusion()
    images_ori = svg.denoising_process(
    svg.args.init_image,
    num_frames = svg.args.num_frames,
    prompts=[svg.args.prompt] * svg.args.batch_size,
    num_inference_steps=svg.args.num_inference_steps, 
    downscale=svg.args.downscale,
    threshold=svg.args.threshold,
    resample=svg.args.resample,
    num_views=svg.args.num_views,
    negative_prompt=svg.args.negative_prompt,
    soft_mask_k=svg.args.soft_mask_k, 
    soft_mask_type=svg.args.soft_mask_type,
    frame_matrix_end=svg.args.frame_matrix_end,  
    update_source_latent=svg.args.update_source_latent, 
    update_source_latent_end=svg.args.update_source_latent_end,
    last_view_idx=svg.args.last_view_idx,
    last_view_resample= svg.args.last_view_resample,
    fix_last_view_begin=svg.args.fix_last_view_begin, 
    stage2_resample=svg.args.stage2_resample, 
    stage2_time_direction_denoise_freq=svg.args.stage2_time_direction_denoise_freq,
    erode_size=svg.args.erode_size,
    middle2leftright=svg.args.middle2leftright,
    monocular_video_idx=svg.args.monocular_video_idx,
    right2left=svg.args.right2left,
    height=svg.args.height,
    width=svg.args.width,
    latent_h=svg.args.latent_h,
    latent_w=svg.args.latent_w,
    ) 
    
    

    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


 




