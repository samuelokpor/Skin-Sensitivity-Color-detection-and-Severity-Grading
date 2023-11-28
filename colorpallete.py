import cv2
import numpy as np
import json
import os
import colour
import torch

from model import UNET
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNET(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("models/redskin1.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Set up the validation data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def infer_red_regions(model, image):
    """
    Applies the trained model on the image to detect red regions.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)

    # Convert the predictions to binary mask
    predictions = torch.sigmoid(predictions)
    mask = (predictions > 0.5).float().squeeze().cpu().numpy()
    
    return mask

def parse_cube_file(cube_file_path):
    # Read the .CUBE file
    with open(cube_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract LUT size from .CUBE file
    size_line = [line for line in lines if "LUT_3D_SIZE" in line][0]
    size = int(size_line.split()[-1])

    # Extract LUT data from .CUBE file
    lut_data = [list(map(float, line.strip().split())) for line in lines if not line.startswith('#') and len(line.strip().split()) == 3]
    lut_data = np.array(lut_data).reshape(size, size, size, 3)

    return lut_data

def invert_lut_data(lut_data):
    """
    Invert all the colors in the LUT data.
    """
    max_value = np.max(lut_data)  # Find the maximum value in the LUT
    inverted_lut_data = max_value - lut_data  # Subtract each value from the maximum value
    return inverted_lut_data

def enhance_red_channel_in_lut(lut_data, scale_factor=1.15):
    """
    Enhance the intensity of the red channel in the LUT data.
    Parameters:
    - lut_data: The LUT data from the .CUBE file.
    - scale_factor: The factor by which to multiply the red channel values.
    Returns:
    - The LUT data with enhanced red channel.
    """
    enhanced_lut_data = lut_data.copy()
    enhanced_lut_data[..., 0] = np.clip(enhanced_lut_data[..., 0] * scale_factor, 0, 1)
    return enhanced_lut_data

def black_to_pink(lut_data, pink_intensity=[1, 0.6, 0.8], threshold=0.3):
    """
    Convert black colors to pink in a LUT.
    The threshold determines how close a color must be to black to be considered black.
    The pink_intensity determines the RGB value of the pink (default is [1, 0.6, 0.8] which is a light shade of pink).
    """
    new_lut = lut_data.copy()
    # Calculate the distance of each RGB value to pure black.
    black_distance = np.linalg.norm(lut_data, axis=-1)
    # If the distance is below the threshold, set the color to pink.
    mask = black_distance < threshold
    new_lut[mask] = pink_intensity
    return new_lut

def enhance_brown(lut_3d_data, enhancement_factor=5):
    """
    Enhance brown colors in the 3D LUT data.
    
    Parameters:
    - lut_3d_data: The loaded 3D LUT data as a numpy array.
    - enhancement_factor: The factor by which to boost the brownish colors. 
      Default is 1.5, but you can adjust to get the desired effect.
      
    Returns:
    - Modified LUT data with enhanced brown colors.
    """
    
    # Create a new LUT data array to hold the enhanced values
    enhanced_lut = np.copy(lut_3d_data)
    
    # Iterate through each entry in the 3D LUT
    for x in range(lut_3d_data.shape[0]):
        for y in range(lut_3d_data.shape[1]):
            for z in range(lut_3d_data.shape[2]):
                # Extract RGB values
                r, g, b = lut_3d_data[x, y, z]
                
                # Check if the current color is approximately brown
                if r > 0.4 and g > 0.3 and b < 0.5:
                    enhanced_lut[x, y, z, 0] = min(r * enhancement_factor, 1)
                    enhanced_lut[x, y, z, 1] = min(g * enhancement_factor, 1)
                    # Keeping blue as it is, but you can modify if needed
    
    return enhanced_lut

def enhance_dark_spots(lut_3d_data, enhancement_factor=0.1, threshold=0.2):
    """
    Enhance (darken) the dark spots in the 3D LUT data.
    
    Parameters:
    - lut_3d_data: The loaded 3D LUT data as a numpy array.
    - enhancement_factor: The factor by which to darken the spots. 
      Values < 1 will darken, values > 1 will lighten. Default is 0.5 (50% darker).
    - threshold: The RGB value below which colors are considered "dark spots".
      Default is 0.1 (consider colors close to black).
      
    Returns:
    - Modified LUT data with enhanced dark spots.
    """
    
    # Create a mask for the dark spots in the LUT data
    dark_mask = np.linalg.norm(lut_3d_data, axis=-1) < threshold
    
    # Apply the mask and darken the dark spots by the enhancement factor
    lut_3d_data[dark_mask] = lut_3d_data[dark_mask] * enhancement_factor
    
    return lut_3d_data

def enhance_white(lut_data, enhancement_factor=1.3, threshold=0.055):
    """
    Enhance white colors in a LUT.
    The threshold determines how close a color must be to white to be considered white.
    The enhancement_factor determines the intensity of the enhancement.
    """
    enhanced_lut = lut_data.copy()

    # Calculate the distance of each RGB value to pure white.
    white_distance = np.linalg.norm(lut_data - [1, 1, 1], axis=-1)

    # If the distance is below the threshold, enhance the color.
    white_mask = white_distance < threshold

    enhanced_lut[white_mask] = np.clip(enhanced_lut[white_mask] * enhancement_factor, 0, 1)

    return enhanced_lut



def load_statistics_from_json(filename):
    """
    Load color statistics from a JSON file.
    """
    with open(filename, 'r') as f:
        stats = json.load(f)
    return stats

def apply_statistics(img, stats):
    """
    Adjusts the color of the image using the provided statistics.
    """
    img = img.astype('float32')
    channels = cv2.split(img)
    adjusted_channels = []

    for i, channel_name in enumerate(['r', 'g', 'b']):
        mean = stats[channel_name]['mean']
        std = stats[channel_name]['std']
        adjusted_channel = ((channels[i] - mean) * std / np.std(channels[i]) + mean)
        adjusted_channels.append(adjusted_channel)

    adjusted_img = cv2.merge(adjusted_channels).clip(0, 255).astype('uint8')
    return adjusted_img

if __name__ == "__main__":
    stats = load_statistics_from_json("source_stats.json")

    # Load the 3D LUT from the .CUBE file
    lut_3d_data = parse_cube_file('cubes/5.cube')
    #inverted_lut_3d_data = invert_lut_data(lut_3d_data)
    enhanced_red_lut_data = enhance_red_channel_in_lut(lut_3d_data, scale_factor=1.2)
    whitened_enhanced_red = enhance_white(enhanced_red_lut_data, enhancement_factor=1.5)
    brown_enhanced_lut_data = enhance_brown(lut_3d_data, enhancement_factor=1.5)
    darkened_brown_spots = enhance_dark_spots(brown_enhanced_lut_data)
    #inverted_lut_data = black_to_pink(enhanced_red_lut_data)
    lut_3d = colour.LUT3D(table=whitened_enhanced_red, name='Skin Pigment')


    image_path = 'data/test/20210114151003_jpzd.jpg'

    image = cv2.imread(image_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagelut = image / 255
    imageb = 255 - image
    
    # Apply the color statistics to the image
    color_stat_image = apply_statistics(image, stats)

    color_stat_image = color_stat_image / 255

    # Apply the 3D LUT
    output_lut = lut_3d.apply(imagelut)

    # Convert back to BGR for visualization with OpenCV
    output_image_bgr = (output_lut * 255).astype(np.uint8)                                                                       
    output_image_bgr = cv2.cvtColor(output_image_bgr, cv2.COLOR_RGB2BGR)
    #output_image_bgr = 255 - output_image_bgr

    ####MODEL inference
    # Infer the red regions from the model
    red_mask = infer_red_regions(model, image)
    # Resize the mask to the original image dimensions
    resized_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    red_overlay = np.zeros_like(image, dtype=np.uint8)
    red_overlay[resized_mask == 1] = [0, 0, 255]  # Red color for the mask


    # Combine the original image and the mask to visualize red regions
    red_visualization = cv2.addWeighted(output_image_bgr, 0.85, red_overlay, 0.15, 0)

    # Display the adjusted image
    cv2.namedWindow("Stats Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stats Image", 900, 900)
    cv2.imshow("Stats Image", color_stat_image)

    # Display the adjusted image
    cv2.namedWindow("LUT on Stats Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LUT on Stats Image", 900, 900)
    cv2.imshow("LUT on Stats Image", output_image_bgr)
    #cv2.imwrite("results/test2.jpg", output_image_bgr)

    # Display the adjusted image
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 900, 900)
    cv2.imshow("Original Image", image)

    # Display the model's prediction
   # Display the model's prediction overlay
    cv2.namedWindow("Model Preds Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model Preds Overlay", 900, 900)
    cv2.imshow("Model Preds Overlay", red_visualization)
    #cv2.imwrite('results/outs2.jpg', red_visualization)

    # Convert the mask to BGR for visualization
    mask_bgr = cv2.cvtColor(resized_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    # Create a blank image with the same dimensions as the resized mask
    red_mask = np.zeros_like(mask_bgr)

    # Set the red channel to the mask's intensity where the mask is 1 (positive detection)
    red_mask[resized_mask == 1] = [0, 0, 255]  # BGR format, so 255 is set to the red channel

    # Display only the model prediction mask
    cv2.namedWindow("Model Preds Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Model Preds Mask", 900, 900)
    cv2.imshow("Model Preds Mask", red_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()