# VLM-Guided-Desnowing-for-Enhanced-Perception-in-Autonomous-Driving
## Team 17 
"Autonomous driving technologies have revolutionized transportation by improving safety and efficiency. However, bad weather conditions, such as heavy snow, continue to pose significant challenges to vehicle perception systems. Snow-covered roads reduce visibility, complicate decision-making, and jeopardize safety. Addressing these issues is crucial for ensuring robust performance in diverse environmental scenarios.


While existing desnowing methods and image enhancement techniques have shown promise in improving visibility, they often lack the precision to focus on specific regions of interest, such as snow-covered roads. Moreover, most techniques operate on the entire image, leading to inefficiencies and unnecessary processing of irrelevant areas.
To address these challenges, we propose a novel approach leveraging Vision-Language Models (VLMs) for targeted segmentation and state-of-the-art desnowing networks. By utilizing the VLM's capability to understand and isolate snow-covered road regions, our method focuses enhancement efforts exclusively on the affected areas, ensuring efficiency and improved visual clarity. Specifically, the segmented region is cropped into smaller patches (e.g., 500x500 pixels) for processing with a pretrained desnowing model.


Our workflow begins with using VLMs, such as Segment Anything, to segment snow-covered road regions from input images. These regions are then cropped and passed to a pretrained desnowing model to remove the snow. Finally, advanced image enhancement techniques, including edge filtering and super-resolution, are applied to improve the perceptual quality of the desnowed images, optimizing PSNR and other quality metrics.
Our approach offers three major contributions:

(1) a novel application of VLMs for precise region-based desnowing in autonomous driving scenarios,

(2) integration of state-of-the-art desnowing networks to enhance targeted regions

(3) post-processing with edge enhancement and super-resolution to improve the visual quality and PSNR of the enhanced images. Together, these advancements ensure more reliable perception and decision-making in adverse weather conditions.
By focusing on efficiency and precision, our method represents a significant step forward in enabling autonomous vehicles to navigate safely in snowy conditions, paving the way for robust all-weather autonomous systems."


