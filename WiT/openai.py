import os
import openai

# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = "Please input your API"
# from https://github.com/popjane/free_chatgpt_api?tab=readme-ov-file#%E5%B8%B8%E7%94%A8%E5%BA%94%E7%94%A8%E6%94%AF%E6%8C%81

# all client options can be configured just like the `OpenAI` instantiation counterpart
openai.base_url = "https://free.v36.cm/v1/"
openai.default_headers = {"x-foo": "true"}

completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "can you tell where is the opposite lane in this picture ? ",
        },
    ],
)
print(completion.choices[0].message.content)

# Result :From the image, the opposite lane can be observed on the left side of the image, as evidenced by the faint visibility of the road markings separating the traffic lanes. The markings indicate that the left side is designated for vehicles traveling in the opposite direction, though the heavy fog makes it less distinct.

# from PIL import Image
# VLM generate cropping code : 
from PIL import Image

# Load the uploaded image
image_path = "/mnt/data/shutterstock_355284197-1024x683.jpg"
image = Image.open(image_path)

# Define the approximate region for the opposite lane in pixels
# Assuming the opposite lane is towards the left, estimate the crop area
# Adjusting based on visual approximation, this region might need refinement
width, height = image.size
crop_region = (0, 0, width // 2, height)  # Crop left half of the image

# Crop the image
cropped_image = image.crop(crop_region)

# Save the cropped image
cropped_image_path = "/mnt/data/cropped_opposite_lane.jpg"
cropped_image.save(cropped_image_path)

cropped_image_path