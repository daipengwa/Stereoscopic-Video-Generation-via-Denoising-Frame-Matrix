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

# current video generation model: zeroscope, 1/8 downsample. 

# leftview2rightview
python create_frame_matrix.py --img_path=./Videos/An_astronaut_in_full_space_suit_riding_a_horse_320x576/ --depth_path=./Videos/An_astronaut_in_full_space_suit_riding_a_horse_320x576/ --output_path=./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320 --max_baseline 0.07 --num_training_views 8  --left2right
python svg.py --prompt 'An astronaut in full space suit riding a horse' --init_image ./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse --output_root ./results/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320  --latent_w 72 --latent_h 40

# # rightview2leftview
# python create_frame_matrix.py --img_path=./Videos/An_astronaut_in_full_space_suit_riding_a_horse_320x576/ --depth_path=./Videos/An_astronaut_in_full_space_suit_riding_a_horse_320x576/ --output_path=./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320 --max_baseline 0.07 --num_training_views 8  --right2left
# python svg.py --prompt 'An astronaut in full space suit riding a horse' --init_image ./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse --output_root ./results/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320  --latent_w 72 --latent_h 40 --right2left

# # middle2leftright (best perfromace with a liitle more time)
# python create_frame_matrix.py --img_path=./Videos/An_astronaut_in_full_space_suit_riding_a_horse_320x576/ --depth_path=./Videos/An_astronaut_in_full_space_suit_riding_a_horse_320x576/ --output_path=./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320 --max_baseline 0.07 --num_training_views 8  --middle2leftright
# python svg.py --prompt 'An astronaut in full space suit riding a horse' --init_image ./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse --output_root ./results/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320  --latent_w 72 --latent_h 40 --middle2leftright

# # multi-view generation
# python svg.py --prompt 'An astronaut in full space suit riding a horse' --init_image ./Videos/frame_matrix/An_astronaut_in_full_space_suit_riding_a_horse --output_root ./results/An_astronaut_in_full_space_suit_riding_a_horse/ --num_frames 16 --width 576 --height 320  --latent_w 72 --latent_h 40 --frame_matrix_end 0 --fix_last_view_begin 600


