import cv2
import torch
import numpy as np
from torchvision import transforms
from resnet import resnet34
import os
import json
import colour
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'models/bestmodel5a.pth'  # Update this path
model = resnet34(num_classes=1)  # Use the same model architecture
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Assuming the maximum possible severity score is 100
max_severity_score = 100.0

# Define the transformation for test data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load and preprocess the test data
def load_test_data(test_dir):
    test_data = []
    json_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(test_dir, json_file)
        with open(json_path, 'r') as file:
            annotation_data = json.load(file)
        scores = [int(shape['label']) for shape in annotation_data['shapes'] if shape['label'].isdigit()]
        severity_score = sum(scores) / len(scores) if scores else 0

        # Check for corresponding image file with .jpg, .jpeg, or .png extension
        base_file_name = json_file.rsplit('.', 1)[0]
        image_file = find_image_file(test_dir, base_file_name)

        if image_file:
            image_path = os.path.join(test_dir, image_file)
            test_data.append((image_path, severity_score))
    return test_data

def find_image_file(folder, base_name):
    for ext in ['.jpg', '.jpeg', '.png']:
        full_path = os.path.join(folder, base_name + ext)
        if os.path.isfile(full_path):
            return base_name + ext
    return None

# Define transformation for test data (without PIL)
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0, 1]
    image = image.transpose(2, 0, 1)  # Change to CHW format
    image = torch.tensor(image, dtype=torch.float32)
    return image.unsqueeze(0)

# Perform inference
def infer(model, image):
    image = image.to(device)
    with torch.no_grad():
        prediction = model(image)
        prediction = prediction.item()  # Get the scalar value
    return prediction

# Apply LUT using colour
def apply_lut(image, lut):
    # Ensure image is in RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1] range
    image_normalized = image_rgb / 255.0
    # Apply the LUT
    image_lut_applied = colour.LUT3D.apply(lut, image_normalized)
    # Convert back to 8-bit BGR format for OpenCV
    image_lut_applied_bgr = (image_lut_applied * 255).astype(np.uint8)
    image_lut_applied_bgr = cv2.cvtColor(image_lut_applied_bgr, cv2.COLOR_RGB2BGR)
    return image_lut_applied_bgr #  Convert back to [0, 255] range


# # Visualization Function
# def visualize(original_image, processed_image, ground_truth, predicted_score):
#     # Display the original image
#     cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Image", 900, 900)
#     cv2.imshow("Original Image", original_image)
    
#     # Display the processed image
#     cv2.namedWindow("LUT Image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("LUT Image", 900, 900)
#     cv2.imshow("LUT Image", processed_image)

#     print(f"Ground Truth Severity Score: {ground_truth}, Predicted Severity Score: {predicted_score:.2f}")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Visualization Function
# def visualize(original_image, processed_image, ground_truth, predicted_score):
#     # Prepare text for overlay
#     ground_truth_text = f"Ground Truth: {ground_truth}"
#     predicted_score_text = f"Predicted: {predicted_score:.2f}"

#     # Set font parameters
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.5
#     font_color = (255, 255, 255)  # White color
#     line_type = 2

#     # Add text to original image
#     cv2.putText(original_image, ground_truth_text, (10, 30), font, font_scale, font_color, line_type)
#     cv2.putText(original_image, predicted_score_text, (10, 60), font, font_scale, font_color, line_type)

#     # Display the original image with text overlay
#     cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Original Image", 900, 900)
#     cv2.imshow("Original Image", original_image)
    
#     # Display the processed image
#     cv2.namedWindow("LUT Image", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("LUT Image", 900, 900)
#     cv2.imshow("LUT Image", processed_image)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# Visualization Function
def visualize(original_image, processed_image, ground_truth, predicted_score):
    # Prepare titles with ground truth and predicted score
    original_title = f"Original Image - Ground Truth: {ground_truth}"
    processed_title = f"LUT Image - Predicted: {predicted_score:.2f}"

    # Plotting
    plt.figure(figsize=(12, 6))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.title(original_title)
    plt.axis('off')  # Turn off axis numbers

    # Display the processed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.title(processed_title)
    plt.axis('off')  # Turn off axis numbers

    plt.show()



# Test directory
test_dir = 'data/test'  # Update this path
test_data = load_test_data(test_dir)

# Read .CUBE LUT using colour library
cube_file_path = 'cubes/5.cube'
lut = colour.read_LUT(cube_file_path)

# # Predict and visualize
# for image_path, ground_truth in test_data:
#     original_image = cv2.imread(image_path)
#     original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     preprocessed_image = preprocess_image(original_image_rgb)
#     predicted_score = infer(model, preprocessed_image)

#     # Re-normalize the predicted score
#     predicted_score *= max_severity_score

#     # Apply LUT
#     processed_image = apply_lut(original_image, lut)

#     # Visualize
#     visualize(original_image, processed_image, ground_truth, predicted_score)


# Predict and visualize
for image_path, ground_truth in test_data:
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    preprocessed_image = preprocess_image(original_image_rgb)
    predicted_score = infer(model, preprocessed_image)

    # Re-normalize the predicted score
    predicted_score *= max_severity_score

    # Apply LUT
    processed_image = apply_lut(original_image, lut)

    # Visualize
    visualize(original_image, processed_image, ground_truth, predicted_score)



