# SVG

**SVG: 3D Stereoscopic Video Generation via Denoising Frame Matrix**  
[Peng Dai], [Feitong Tan*], [Qiangeng Xu*], [David Futschik], [Ruofei Du], [Sean Fanello], [Xiaojuan Qi], [Yinda Zhang]
<br>[Paper](https://arxiv.org/pdf/2407.00367), [Project_page](https://daipengwa.github.io/SVG_ProjectPage/)

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).


## Environment
```
conda create -n svg python=3.8
conda activate svg

# we use torch 2.4, other versions still work
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```


## Data preprocessing

* Please download [datasets](https://drive.google.com/drive/folders/1qu5Z6U505MBidjNR2p77G9bmUy-eYND9?usp=sharing) used in this paper. The layout looks like this:
```
SVG
├── Videos
    │──An_astronaut_in_full_space_suit_riding_a_horse_320x576 
    │   │──images
    │   │   |──00000.jpg
    │   │   |──00001.jpg
    │──Obama_is_speaking_320x576 
    │   │──images
    │   │   |──00000.jpg
    │   │   |──00001.jpg
 
```

* __Depth prediction.__ Predict depth for each frame. You need to run under [depthanything_v1](https://github.com/LiheYoung/Depth-Anything) environment.
```Shell
cd third_party/Depth-Anything/
python svg_run.py --encoder vitl --img-path ../../Videos/Obama_is_speaking_320x576/images/ --outdir ../../Videos/Obama_is_speaking_320x576/depths/
```

* __Depth stabilization.__ We stabilize the depth changes along the time axis. You need to run under [RAFT](https://github.com/princeton-vl/RAFT) environment and install guidedfilter by running `pip install opencv-contrib-python`.  
```Shell
cd third_party/RAFT/
python svg_enhance_depth.py --path=../../Videos/Obama_is_speaking_320x576/images/  --depth_path=../../Videos/Obama_is_speaking_320x576/depths/ --out_dir=../../Videos/Obama_is_speaking_320x576/depths_refined/
```


## Run video generation
* __Construct frame matrix__
```Shell
python create_frame_matrix.py --img_path=./Videos/Obama_is_speaking_320x576/ --depth_path=./Videos/Obama_is_speaking_320x576/ --output_path=./Videos/frame_matrix/Obama_is_speaking/  --num_frames 16 --width 576 --height 320 --max_baseline 0.07 --num_training_views 8  --left2right
```

* __Stereoscopic video generation__
```Shell
python svg.py --prompt 'Obama is speaking' --init_image ./Videos/frame_matrix/Obama_is_speaking --output_root ./results/Obama_is_speaking/  --num_frames 16 --width 576 --height 320  --latent_w 72 --latent_h 40
```

* __Multi-view video generation__
```Shell
python svg.py --prompt 'Obama is speaking' --init_image ./Videos/frame_matrix/Obama_is_speaking --output_root ./results/Obama_is_speaking/ --num_frames 16 --width 576 --height 320  --latent_w 72 --latent_h 40 --frame_matrix_end 0 --fix_last_view_begin 600
```

More useful commands, please refer to 'run.sh'.

<!-- ## Lift to 4D

## Editing

## Solve issues in depth-based warping -->

## Tips
### Depth range
We use estimated relative depth and normalize it into 1~10m. Note that the closer the foreground content is, the more disoccluded areas need to be inpainted, making the task more challenging. You can modify the depth values to achieve stereoscopic effects you prefer. Alternatively, metric depth is a good choice if avaiable.

### Camera settings
We use two parallel cameras (fx, fy: 800) with a 7cm distance between them (objects will be in front of the screen). According to your preference, you can change the fx/fy or toe in the two cameras (two cameras have a coverage point).

### The number of cameras between left and right views
We place 8 cameras between left and right views. You can reduce the number of cameras to expedite the generation process when the case is easy or the disoccluded regions are small. In practice, we found that 4 cameras also produce competitive results.

### Video generation model
The current implementation is based on a text2video model (i.e., zeroscope) and therefore requires text prompts as inputs, which have an influence on the final results.

## Citation
Please consider staring this repository and citing the following paper if you feel this repository useful.

```
@inproceedings{
dai2025svg,
title={{SVG}: 3D Stereoscopic Video Generation via Denoising Frame Matrix},
author={Peng Dai and Feitong Tan and Qiangeng Xu and David Futschik and Ruofei Du and Sean Fanello and XIAOJUAN QI and Yinda Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=sx2jXZuhIx}
}
```


## Contact
If you have any questions, you can email me (daipengwa@gmail.com). 
