# import os
# import json
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from PIL import Image
# import torch

# # Check if CUDA is available
# if torch.cuda.is_available():
#     print("CUDA is available")
# else:
#     print("CUDA is not available")

# # Directory settings
# image_folder = 'data/train'  # Set to the path where your images are stored
# json_folder = 'data/train'   # Set to the path where your JSON files are stored

# # Function to convert JSON annotations to mask
# def json_to_mask(json_path, img_shape):
#     with open(json_path, "r") as read_file:
#         data = json.load(read_file)
#     mask = np.zeros(img_shape[:2], dtype=np.uint8)
#     for shape in data['shapes']:
#         points = np.array(shape['points'], dtype=np.int32)
#         cv2.fillPoly(mask, [points], 255)
#     return mask

# class RedAreasSegmentationDataset(Dataset):
#     def __init__(self, image_folder, json_folder, transform=None):
#         self.image_folder = image_folder
#         self.json_folder = json_folder
#         self.transform = transform
#         self.json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

#     def __len__(self):
#         return len(self.json_files)

#     def __getitem__(self, idx):
#         json_name = self.json_files[idx]
#         json_path = os.path.join(self.json_folder, json_name)
#         img_name = json_name.split('.')[0] + '.jpg'
#         img_path = os.path.join(self.image_folder, img_name)

#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mask = json_to_mask(json_path, img.shape)

#         severity_scores = self.extract_severity_scores(json_path)
#         mean_severity_score = np.mean(severity_scores) if severity_scores else 0

#         if self.transform:
#             img = self.transform(img)
#             mask = self.transform(mask)
        
#         return img, mask, mean_severity_score

#     def extract_severity_scores(self, json_path):
#         with open(json_path, "r") as read_file:
#             data = json.load(read_file)
#         scores = [int(shape['label']) for shape in data['shapes'] if shape['label'].isdigit()]
#         return scores

# # Transformation to resize images and masks and convert them to torch tensors
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Initialize dataset and dataloader
# dataset = RedAreasSegmentationDataset(image_folder=image_folder, json_folder=json_folder, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Visualization
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
# for i, (batch_images, batch_masks, batch_scores) in enumerate(dataloader):
#     if i >= 2:
#         break
#     for j in range(2):
#         ax[i, j*2].imshow(batch_images[j].permute(1, 2, 0))
#         ax[i, j*2].set_title(f'Image - Severity Score: {batch_scores[j]}')
#         ax[i, j*2+1].imshow(batch_masks[j][0], cmap='gray')
#         ax[i, j*2+1].set_title('Mask')

# plt.show()


import os
import json
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch

class RedAreasDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_severity_score=100.0):
        self.image_dir = image_dir
        self.transform = transform
        self.max_severity_score = max_severity_score 
        self.data_entries = self._load_annotations()

    def _load_annotations(self):
        data_entries = []
        json_files = [f for f in os.listdir(self.image_dir) if f.endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(self.image_dir, json_file)
            with open(json_path, 'r') as file:
                annotation_data = json.load(file)

            scores = [int(shape['label']) for shape in annotation_data['shapes'] if shape['label'].isdigit()]
            severity_score = sum(scores) / len(scores) if scores else 0
            mask_points = [shape['points'] for shape in annotation_data['shapes'] if shape['label'].isdigit()]

            # Check for corresponding image file with .jpg, .jpeg, or .png extension
            base_file_name = json_file.rsplit('.', 1)[0]
            image_file = self._find_image_file(base_file_name)

            if image_file:
                data_entries.append({
                    'file_path': os.path.join(self.image_dir, image_file),
                    'severity_score': severity_score,
                    'mask_points': mask_points
                })

        return data_entries

    def _find_image_file(self, base_file_name):
        for extension in ['.jpg', '.jpeg', '.png']:
            potential_file = f'{base_file_name}{extension}'
            if os.path.exists(os.path.join(self.image_dir, potential_file)):
                return potential_file
        return None

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        image = Image.open(entry['file_path']).convert('RGB')

        # Create mask
        mask = Image.new('L', image.size, 0)
        for points in entry['mask_points']:
            flattened_points = [tuple(point) for point in points]
            ImageDraw.Draw(mask).polygon(flattened_points, outline=1, fill=1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Normalize severity score
        normalized_score = entry['severity_score'] / self.max_severity_score

        return image, mask, normalized_score

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image_dir = 'data/train/'
dataset = RedAreasDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Visualization
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
for i, (batch_images, batch_masks, batch_scores) in enumerate(dataloader):
    if i >= 2:
        break
    for j in range(2):
        ax[i, j*2].imshow(batch_images[j].permute(1, 2, 0))
        ax[i, j*2].set_title(f'Image - Severity Score: {batch_scores[j]}')

        # Ensure the mask has two dimensions
        mask = batch_masks[j].squeeze()  # Remove any singleton dimensions
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)  # Add a dimension if needed

        ax[i, j*2+1].imshow(mask, cmap='gray')
        ax[i, j*2+1].set_title('Mask')

plt.show()
