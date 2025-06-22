# Python snippet
from PIL import Image
import numpy as np

# Load
img = np.array(Image.open('input.jpg').convert('RGB'))
out = np.array(Image.open('output.jpg').convert('RGB'))

# Count unique colors
print("Input unique colors:", len(np.unique(img.reshape(-1, 3), axis=0)))
print("Output unique colors:", len(np.unique(out.reshape(-1, 3), axis=0)))
