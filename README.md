

# **VLM-Guided Desnowing for Enhanced Perception in Autonomous Driving**  
**Team 17**  
Final Project for CV_2024  

---

## **Overview**  
Autonomous driving technologies have brought revolutionary advancements in transportation by enhancing safety and efficiency. However, adverse weather conditions, such as heavy snow, present a significant challenge for vehicle perception systems. Snow-covered roads reduce visibility, complicate decision-making, and compromise safety.  

This repository presents our novel approach, leveraging Vision-Language Models (VLMs) and state-of-the-art desnowing techniques, to address the challenges of snow-obscured perception in autonomous driving. Our approach focuses on targeted desnowing, improving both efficiency and accuracy in detecting critical objects and hazards.

---

## **Key Contributions**  

1. **Vision-Language Models (VLMs)** for precise segmentation of snow-covered regions (e.g., Segment Anything and ViLD).  
2. Integration of **state-of-the-art desnowing models** to remove snow from targeted regions while preserving image details.  
3. Post-processing techniques, including **edge enhancement** and **super-resolution**, to optimize image quality and improve metrics like PSNR.  

Together, these innovations ensure robust perception and decision-making for autonomous vehicles under snowy conditions.

---

## **Features**  

- **Targeted Snow Region Detection**: Efficient segmentation and cropping of snow-covered road regions using VLMs.  
- **Pretrained Desnowing Models**: Removal of snow obstructions with advanced desnowing networks.  
- **Enhanced Image Quality**: Application of edge filtering and super-resolution to enhance clarity and resolution.  
- **Pipeline Flexibility**: Adaptable to new scenarios with minimal adjustments, thanks to the open-vocabulary detection of VLMs.  

---

## **Installation**  

### **Clone the Repository**  
```bash
git clone https://github.com/your-username/VLM-Guided-Desnowing.git
cd VLM-Guided-Desnowing
```




## **Repository Structure**  

```
.
├── add_snow_mask.py          # Segment snow-covered regions
├── calculate_bbox_size.py    # Calculate bounding box sizes
├── cv-final.yaml             # Environment configuration
├── experiment.py             # Main experiment code for desnowing
├── experiment_psnr.py        # Evaluate image quality (PSNR, etc.)
├── preprocess_data.py        # Resize and preprocess input images
├── resize_img.py             # Additional resizing utilities
├── sample_data.py            # Sample data generator for testing
├── visualize_result.py       # Visualize the results of desnowing
├── datasets/                 # Dataset directory (to be added)
└── models/                   # Pretrained models directory
```

---

## **Results**  

Our method demonstrates significant improvements in:  
- **PSNR and visual clarity** of desnowed images.  
- Accurate detection of critical objects, such as vehicles and pedestrians, under snowy conditions.  
- Enhanced robustness and efficiency for autonomous driving applications in adverse weather.  

### Sample Visualization  
- Original Image → Snow Obstructed → Desnowed Image  
- Improved object detection and hazard identification.

---

## **Acknowledgements**  
This work builds on the following:  
- [ViLD: Vision-Language Knowledge Distillation](https://arxiv.org/abs/2111.09883)  
- Pretrained desnowing models from ICCV2021-HDCWNet.  


