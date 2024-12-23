import os
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import subprocess
import argparse

def get_filename_without_extension(filepath):
    """Get filename without extension from filepath"""
    return os.path.splitext(os.path.basename(filepath))[0]

def run_remove_anything(image_path, point_coords, sam_checkpoint="./pretrained_models/sam_vit_h_4b8939.pth"):
    """Run remove_anything.py to get the background"""
    command = [
        "python", "remove_anything.py",
        "--input_img", image_path,
        "--coords_type", "key_in",
        "--point_coords", str(point_coords[0]), str(point_coords[1]),
        "--point_labels", "1",
        "--dilate_kernel_size", "15",
        "--output_dir", "./results",
        "--sam_model_type", "vit_h",
        "--sam_ckpt", sam_checkpoint,
        "--lama_config", "./lama/configs/prediction/default.yaml",
        "--lama_ckpt", "./pretrained_models/big-lama"
    ]
    subprocess.run(command)

def get_person_mask(predictor, image, point_coords, point_labels):
    """Get person mask using SAM"""
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    return masks[2]  # Using the third mask as it's typically the most confident

def combine_person_and_background(person_image, background_image, mask):
    """Combine the person from person_image with the background"""
    if isinstance(person_image, Image.Image):
        person_image = np.array(person_image)
    if isinstance(background_image, Image.Image):
        background_image = np.array(background_image)
    
    # Ensure mask is binary and has the right shape
    mask = mask.astype(np.float32)
    if len(mask.shape) == 2:
        mask = np.stack([mask] * 3, axis=-1)
    
    # Combine person and background
    # Use mask to keep the person from person_image and background from background_image
    result = background_image * (1 - mask) + person_image * mask
    return result.astype(np.uint8)

def fix_background(image1_path, image2_path, point_coords, output_path):
    """Main function to fix the background"""
    # First run remove_anything.py on image1 to get the clean background
    run_remove_anything(image1_path, point_coords)
    
    # Get the base filename from image1_path
    image1_filename = get_filename_without_extension(image1_path)
    
    # Construct the correct path to inpainted_with_mask_2.png
    background_path = os.path.join("./results", image1_filename, "inpainted_with_mask_2.png")
    
    # Load the clean background
    clean_background = cv2.imread(background_path)
    if clean_background is None:
        raise FileNotFoundError(f"Could not find or load background image at {background_path}")
    clean_background = cv2.cvtColor(clean_background, cv2.COLOR_BGR2RGB)
    
    # Load the image with the person
    person_image = cv2.imread(image2_path)
    person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    
    # Ensure images are the same size
    if clean_background.shape != person_image.shape:
        clean_background = cv2.resize(clean_background, 
                                    (person_image.shape[1], person_image.shape[0]))
    
    # Setup SAM for getting the person mask
    predictor = setup_sam()
    
    # Get person mask
    point_labels = np.array([1])  # 1 for foreground
    point_coords = np.array([point_coords])
    mask = get_person_mask(predictor, person_image, point_coords, point_labels)
    
    # Combine person and clean background
    result = combine_person_and_background(person_image, clean_background, mask)
    
    # Save result
    result_pil = Image.fromarray(result)
    result_pil.save(output_path)
    return result

def setup_sam(sam_checkpoint="./pretrained_models/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    """Initialize SAM model"""
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Fix background of an image.")
    parser.add_argument("--image1_path", type=str, required=True, help="Path to the original image with a good background.")
    parser.add_argument("--image2_path", type=str, required=True, help="Path to the processed image with a problematic background.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output fixed image.")
    parser.add_argument("--point_coords", type=int, nargs=2, required=True, help="Coordinates [x, y] to click on the person/foreground.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Process the image
    result = fix_background(
        image1_path=args.image1_path,
        image2_path=args.image2_path,
        point_coords=args.point_coords,
        output_path=args.output_path
    )
