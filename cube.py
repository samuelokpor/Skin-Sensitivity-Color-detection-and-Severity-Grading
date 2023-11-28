# import numpy as np
# import imageio
# import colour
# import cv2

# # Load your image using imageio
# image_path = 'data/test/random.jpg'
# image = imageio.imread(image_path)

# # Convert the image to grayscale using OpenCV
# gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# # Convert the grayscale image back to RGB by repeating the grayscale values across three channels
# rgb_gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# # Normalize the image to [0, 1] range
# image_lut = image / 255.0


# # Read the .CUBE LUT
# cube_file_path = 'cubes/5.cube'
# lut = colour.read_LUT(cube_file_path, format='CUBE')

# # Apply the LUT to your RGB grayscale image
# image_after_lut = colour.LUT3D.apply(lut, image_lut)

# # Convert the image back to 8-bit values for display
# result_image = (image_after_lut * 255.0).astype(np.uint8)
# results_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

# # Display the resulting image using OpenCV
# cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original Image", 900, 900)
# cv2.imshow("Original Image", gray_image ) 

# cv2.namedWindow("LUT Applied Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("LUT Applied Image", 900, 900)
# cv2.imshow("LUT Applied Image", results_bgr ) 
# cv2.imwrite("results/random.jpg", results_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
# import colour

# # Load your image using OpenCV
# image_path = 'data/test/random.jpg'
# original_image = cv2.imread(image_path)

# # Normalize the image to [0, 1] range
# image_lut = original_image / 255.0

# # Read the .CUBE LUT
# cube_file_path = 'cubes/5.cube'
# lut = colour.read_LUT(cube_file_path, format='CUBE')

# # Apply the LUT to the original image
# image_after_lut = colour.LUT3D.apply(lut, image_lut)

# # Convert the LUT-applied image back to 8-bit values for display
# result_image = (image_after_lut * 255.0).astype(np.uint8)

# # Display the original image using OpenCV
# cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original Image", 900, 900)
# cv2.imshow("Original Image", original_image)


# # Display the LUT-applied image using OpenCV
# cv2.namedWindow("LUT Applied Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("LUT Applied Image", 900, 900)
# cv2.imshow("LUT Applied Image", result_image)

# # Wait for a key press and close windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save the LUT-applied image if needed
# cv2.imwrite("results/random_lut_applied.jpg", result_image)


import cv2
import colour
import numpy as np

# Load your image using OpenCV
image_path = 'data/test/44115c1ssk00044.jpg'
original_image_bgr = cv2.imread(image_path)

# Convert BGR to RGB for LUT application
original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

# Normalize the image to [0, 1] range
image_lut_normalized = original_image_rgb / 255.0

# Read the .CUBE LUT
cube_file_path = 'cubes/5.cube'
lut = colour.read_LUT(cube_file_path, format='CUBE')

# Apply the LUT to the normalized RGB image
image_after_lut = colour.LUT3D.apply(lut, image_lut_normalized)

# Convert the LUT-applied image back to 8-bit values and then to BGR for display
result_image_rgb = (image_after_lut * 255.0).astype(np.uint8)
result_image_bgr = cv2.cvtColor(result_image_rgb, cv2.COLOR_RGB2BGR)

# Display the original image using OpenCV
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 900, 900)
cv2.imshow("Original Image", original_image_bgr)

# Display the LUT-applied image using OpenCV
cv2.namedWindow("LUT Applied Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LUT Applied Image", 900, 900)
cv2.imshow("LUT Applied Image", result_image_bgr)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the LUT-applied image if needed
cv2.imwrite("results/random_lut_applied.jpg", result_image_bgr)
