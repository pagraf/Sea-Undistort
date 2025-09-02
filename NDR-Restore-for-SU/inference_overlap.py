import os
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import options.options as option
from utils import util
from models import create_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', required=True)
    parser.add_argument('-input_folder', type=str, help='Path to folder with input images.', required=True)
    parser.add_argument('-output_folder', type=str, help='Path to folder where output will be saved.', required=True)
    args = parser.parse_args()
    
    # Load options
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    
    # Initialize random seed
    seed = opt['train'].get('manual_seed', None)
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    
    torch.backends.cudnn.benchmark = True

    resume_state = None
    
    # Create model
    model = create_model(opt)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Configurable parameters for patch-based processing
    # patch_size: Maximum patch size for processing (increase for better quality, decrease for lower memory usage)
    # overlap: Overlap between patches in pixels (prevents artifacts at patch boundaries)
    # step: Step size ensuring patches overlap
    patch_size = 1024
    overlap = 128
    step = patch_size - overlap
    
    # Process all images in input folder
    for file_name in os.listdir(args.input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            input_path = os.path.join(args.input_folder, file_name)
            print(f"Processing: {input_path}")
            
            # Load image, convert to RGB and normalize
            img = Image.open(input_path).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0  # Values in [0, 1]
            # (H, W, C) -> (C, H, W)
            img_np = img_np.transpose(2, 0, 1)
            C, H, W = img_np.shape
            
            # Convert input image to 4D tensor
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            # Create output buffer and weight matrix
            output_tensor = torch.zeros((C, H, W), dtype=torch.float32, device=img_tensor.device)
            weight_tensor = torch.zeros((1, H, W), dtype=torch.float32, device=img_tensor.device)
            
            # Divide image into overlapping patches
            for y in range(0, H, step):
                for x in range(0, W, step):
                    y_end = min(y + patch_size, H)
                    x_end = min(x + patch_size, W)
                    
                    # Extract current patch
                    patch = img_tensor[:, :, y:y_end, x:x_end]
                    _, _, ph, pw = patch.shape
                    
                    # Add padding if ph or pw are not divisible by 8
                    pad_h = 8 - ph % 8 if ph % 8 != 0 else 0
                    pad_w = 8 - pw % 8 if pw % 8 != 0 else 0
                    if pad_h or pad_w:
                        patch_pad = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                    else:
                        patch_pad = patch
                        
                    # Run inference on patch
                    demo_data = {'lq': patch_pad, 'name_lq': [f'{file_name}_patch_{y}_{x}']}
                    model.feed_data_demo(demo_data)
                    model.test()
                    visuals = model.get_current_visuals_demo()
                    out_patch = visuals['out_img']
                    # Remove additional padding to return to original patch size
                    out_patch = out_patch[:C, :ph, :pw]
                    
                    # Weight matrix: uniform value; will be averaged later in overlap regions
                    weight_mask = torch.ones((1, ph, pw), device=img_tensor.device, dtype=torch.float32)
                    
                    # Accumulate results
                    output_tensor[:, y:y_end, x:x_end] += out_patch
                    weight_tensor[:, y:y_end, x:x_end] += weight_mask
                    
                    # CUDA memory cleanup if needed
                    torch.cuda.empty_cache()
            
            # Average overlapping regions
            weight_tensor[weight_tensor == 0] = 1.0  # Avoid division by zero
            output_tensor /= weight_tensor
            
            # Convert result to numpy array
            out_img = output_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
            # Note: If needed, reverse color channels (e.g., from BGR to RGB): out_img = out_img[::-1, :, :]
            out_img = out_img.transpose(1, 2, 0)  # (H, W, C)
            out_img = np.clip(out_img, 0, 1)
            
            # Save result
            output_image = Image.fromarray((out_img * 255).astype(np.uint8))
            output_path = os.path.join(args.output_folder, file_name)
            output_image.save(output_path)
            print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
