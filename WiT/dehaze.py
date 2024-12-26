import image_dehazer
import cv2

# Load the input image (must be a color image)
HazeImg = cv2.imread('dsc_0066_11-2003pse.jpg')

# Check if the image was loaded successfully
if HazeImg is None:
    raise ValueError("Image not found or unable to load.")

# Remove haze
HazeCorrectedImg, HazeTransmissionMap = image_dehazer.remove_haze(HazeImg)

# Save the images to disk
cv2.imwrite('hazy_image.jpg', HazeImg)                 # Save the original hazy image
cv2.imwrite('enhanced_image.jpg', HazeCorrectedImg)    # Save the haze-free enhanced image

print("Images saved as 'hazy_image.jpg' and 'enhanced_image.jpg'")
