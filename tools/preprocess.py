#!/usr/bin/env python3
"""
Image preprocessing for YOLOv5 inference
Converts input image to NCHW tensor format
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Resize image to new_shape with letterbox padding
    Returns: resized image, ratio, (dw, dh)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, don't scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(image_path, img_size=640):
    """
    Preprocess image for YOLOv5 inference
    Returns: NCHW tensor (numpy array), metadata dict
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_h, original_w = img.shape[:2]
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Letterbox resize
    img, ratio, (dw, dh) = letterbox(img, new_shape=(img_size, img_size), auto=False)
    
    # Normalize: [0, 255] -> [0.0, 1.0]
    img = img.astype(np.float32) / 255.0
    
    # Convert to NCHW: (H, W, C) -> (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    metadata = {
        "original_shape": list(img.shape),
        "original_image_size": [original_h, original_w],
        "ratio": list(ratio),
        "padding": [float(dw), float(dh)],
        "img_size": img_size
    }
    
    return img, metadata


def save_tensor(tensor, output_path):
    """
    Save tensor to binary file (compatible with C tensor_load)
    Format: [n, c, h, w] (int32_t) + data (float32)
    """
    with open(output_path, 'wb') as f:
        # Write dimensions
        dims = np.array(tensor.shape, dtype=np.int32)
        dims.tofile(f)
        
        # Write data
        tensor.astype(np.float32).tofile(f)


def save_meta_txt(metadata, output_path, image_name):
    """
    Save metadata to text file (human-readable format)
    """
    with open(output_path, 'w') as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Tensor shape: {metadata['original_shape']}\n")
        f.write(f"Image size: {metadata['img_size']}\n")
        f.write(f"Scale ratio: {metadata['ratio']}\n")
        f.write(f"Padding: (dw={metadata['padding'][0]:.2f}, dh={metadata['padding'][1]:.2f})\n")
        f.write(f"\n")
        f.write(f"Format: NCHW (Batch, Channels, Height, Width)\n")
        f.write(f"Data type: float32\n")
        f.write(f"Normalized: [0.0, 1.0]\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess images from data/images folder')
    parser.add_argument('--input-dir', type=str, default='data/images', 
                        help='Input images directory (default: data/images)')
    parser.add_argument('--output-dir', type=str, default='data/inputs', 
                        help='Output directory for preprocessed tensors (default: data/inputs)')
    parser.add_argument('--size', type=int, default=640, help='Input image size')
    parser.add_argument('--image', type=str, default=None, 
                        help='Process single image (if not specified, process all images in input-dir)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Get list of images to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images_to_process = []
    
    if args.image:
        # Process single image
        image_path = input_dir / args.image
        if image_path.exists() and image_path.suffix.lower() in image_extensions:
            images_to_process.append(image_path)
        else:
            print(f"Error: Image '{args.image}' not found in '{args.input_dir}'")
            return 1
    else:
        # Process all images in directory
        for img_path in input_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                images_to_process.append(img_path)
    
    if not images_to_process:
        print(f"No images found in '{args.input_dir}'")
        return 1
    
    print(f"Found {len(images_to_process)} image(s) to process\n")
    
    # Process each image
    for img_path in images_to_process:
        image_name = img_path.stem
        print(f"Processing: {img_path.name}")
        
        try:
            # Preprocess
            tensor, metadata = preprocess_image(str(img_path), args.size)
            
            # Generate output filenames
            bin_path = output_dir / f"{image_name}.bin"
            meta_path = output_dir / f"{image_name}_meta.txt"
            
            # Save tensor
            save_tensor(tensor, str(bin_path))
            print(f"  ✓ Saved tensor to {bin_path}")
            print(f"    Tensor shape: {tensor.shape}")
            
            # Save metadata as text file
            save_meta_txt(metadata, str(meta_path), img_path.name)
            print(f"  ✓ Saved metadata to {meta_path}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}\n")
            continue
    
    print(f"Processing complete! Processed {len(images_to_process)} image(s)")
    return 0


if __name__ == '__main__':
    main()
