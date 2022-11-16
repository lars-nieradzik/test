from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import random
import timm
import os
import cv2
import glob
from PIL import Image

ARCHITECTURE = "efficientnet_b0"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
IMAGE_FOLDER = "images"
EPSILON = 8/255

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_image(file_path):
    im = Image.open(file_path)
    im = im.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
    im = im.convert("RGB")

    return np.asarray(im)

def preprocess_images(images):
    transformer = transforms.Normalize(MEAN, STD)

    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)

    return images

def fgsm_attack(img_path, model, activation_map):
    im_orig = load_image(img_path)
    im_tensor = preprocess_images([im_orig]).to(device)
    im_tensor.requires_grad = True

    out = model(im_tensor)
    pred = torch.argmax(out, dim=1).item()

    loss = F.nll_loss(out, torch.tensor(pred)[None].to(device))
    model.zero_grad()
    loss.backward()

    data_grad = im_tensor.grad.data
    sign_data_grad = data_grad.sign()
    with torch.no_grad():
        perturbed_image = im_tensor.cpu().numpy() + sign_data_grad.cpu().numpy() * activation_map

    std = np.array(STD).reshape((1, 3, 1, 1))
    mean = np.array(MEAN).reshape((1, 3, 1, 1))

    perturbed_image_numpy = 255 * (perturbed_image * std + mean)
    perturbed_image_numpy = np.transpose(perturbed_image_numpy, (0, 2, 3, 1)).astype(np.uint8).squeeze()

    return perturbed_image_numpy

def main():
    print(f"Using device: {device}")

    set_seed(0)

    images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*")))

    model = timm.create_model(ARCHITECTURE, pretrained=True)
    model.eval()
    models = model.to(device)

    attack_successful = 0
    attack_not_successful = 0
    for filename in images:
        resized_image = load_image(filename)
        im_tensor = preprocess_images([resized_image]).to(device)
        with torch.no_grad():
            out = model(im_tensor)
            pred = torch.argmax(out, dim=1).item()

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        activation_map = cv2.Canny(img, threshold1=100, threshold2=200) / 255
        activation_map_resized = cv2.resize(activation_map, (IMAGE_SIZE, IMAGE_SIZE))

        activation_map_resized /= np.sum(activation_map_resized)
        activation_map_resized = activation_map_resized * IMAGE_SIZE * IMAGE_SIZE * EPSILON

        perturbed_image = fgsm_attack(filename, model, activation_map_resized)

        perturbed_image_tensor = preprocess_images([perturbed_image]).to(device)
        with torch.no_grad():
            out = model(perturbed_image_tensor)
            pred2 = torch.argmax(out, dim=1).item()

        if pred != pred2:
            attack_successful += 1
        else:
            attack_not_successful += 1

        fig, axs = plt.subplots(1, 3)

        fig.suptitle(f"Image {filename} => attack {'not' if pred == pred2 else ''} successful")

        axs[0].imshow(resized_image)
        axs[1].imshow(activation_map_resized)
        axs[2].imshow(perturbed_image)

        axs[0].set_title("Original image")
        axs[1].set_title("Activation map (Canny)")
        axs[2].set_title("Perturbed image")

        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')

        plt.show()

    print(f"epsilon F1: {attack_successful / (attack_successful + attack_not_successful)}")

if __name__ == "__main__":
    main()
