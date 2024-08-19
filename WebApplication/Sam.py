import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import numpy as np
from PIL import Image
plt.ion()

def segment(image):

    print("loading model ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_checkpoint = "weight/sam_vit_l_0b3195.pth"
    model_type = "vit_l" 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("masks is generating ===")
    print(f"image shape :{image.shape}")
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print("masks have been generated ===")
    print(masks[0]['segmentation'].shape)
    fig, ax = plt.subplots(figsize=(image.shape[1]/100, image.shape[0]/100), dpi=100)
    ax.imshow(image)
    h = masks[0]['segmentation'].shape[0]
    w = masks[0]['segmentation'].shape[1]
    img = np.ones((h, w, 4))
    img[:, :, 3] = 0
    for mask in masks:
        m = mask['segmentation']
        color = np.concatenate([np.random.random(3), [0.6]])
        img[m] = color
    ax.imshow(img)
    plt.axis('off')
    print("saving image")
    print(os.path.join(os.getcwd(),'static/segment/segment.jpg'))
    plt.savefig(os.path.join(os.getcwd(),'static/segment/segment.jpg'), bbox_inches='tight', pad_inches=0)
    
    image = Image.open(os.path.join(os.getcwd(),'static/segment/segment.jpg'))
    image_ndarray = np.array(image)
    print(image_ndarray.shape)


    return masks





