import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random 
import math
from skimage import io, color, exposure, transform
from PIL import Image

# plot images originales:
image_files = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg"]
fig, axes = plt.subplots(2, len(image_files)//2)
plt.suptitle("Original Images", fontsize=25, fontweight='bold')

for i, image_file in enumerate(image_files):
    image = mpimg.imread(f"Images/{image_file}")
    ax = axes[i % 2, i // 2]
    ax.imshow(image)
    ax.axis('off')
plt.tight_layout()
plt.show()

# plot images resized:
fig, axes = plt.subplots(2, len(image_files)//2)
plt.suptitle("Resized Images", fontsize=25, fontweight='bold')
image_ori = mpimg.imread(f"Images/{image_files[0]}")
images_resized= []
new_size = image_ori.shape
print(new_size)
for i, image_file in enumerate(image_files):
    image = mpimg.imread(f"Images/{image_file}")
    image_resized = transform.resize(image, new_size, anti_aliasing=True)
    images_resized.append(image_resized)

    ax = axes[i % 2, i // 2]
    ax.imshow(image_resized)
    ax.axis('off')
plt.tight_layout()
plt.show()

#
fig, axes = plt.subplots(nrows=2, ncols=len(images_resized)+1, figsize=(48, 8))
ax = axes[0 % 2, 0 // 2]
ax.axis('off')
ax.set_title('Original images:', fontsize=25, va='center', ha='center', fontweight='bold', y=0.5)
ax = axes[1 % 2, 1 // 2]
ax.axis('off')
ax.set_title('Transported images:', fontsize=25, va='center', ha='center', fontweight='bold', y=0.5)
ax = axes[2 % 2, 2 // 2]
ax.set_title("Image to transport", fontsize=25, fontweight='bold')
ax.axis('off')
ax.imshow(image_ori)
ax = axes[3 % 2, 3 // 2]
ax.axis('off')
a = 3

for i, image2_resized in enumerate(images_resized[1:]):
    a+=1

    image1 = image_ori.reshape(-1, 3).copy().astype(float) / 255
    image2 = image2_resized.reshape(-1, 3).copy().astype(float)

    if np.max(image2) > 1:
        image2 /= 255
    ax = axes[a % 2, a // 2]
    ax.imshow(image2_resized)
    ax.axis('off')

    iter = 100
    for i in range(iter):
        # direction:
        d = np.random.random(3)
        d = d / math.sqrt(sum([d[i] ** 2 for i in range(3)]))

        # Projections:
        projected1 = image1.dot(d)
        projected2 = image2.dot(d)

        # Sorting:
        projected1_sort = np.sort(projected1)
        projected2_sort = np.sort(projected2)

        arg_proj1 = np.argsort(projected1)
        arg_proj2 = np.argsort(projected2)

        # update :
        image1[arg_proj1] = image1[arg_proj1] + np.einsum("p,c -> pc", projected2_sort - projected1_sort, d)
    # Affichage résultats:
    a+=1
    ax = axes[a % 2, a // 2]
    ax.imshow(image1.reshape(new_size))
    ax.axis('off')

plt.tight_layout()
plt.show()