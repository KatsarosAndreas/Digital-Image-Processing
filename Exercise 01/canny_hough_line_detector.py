#!/usr/bin/env python3

import pathlib
import numpy as np
from PIL import Image
from skimage import color, feature, transform
import matplotlib.pyplot as plt

# Φόρτωση της εικόνας και μετατροπή σε RGB πίνακα
folder_data = pathlib.Path("Images") / "Ασκηση 6"
image_file = folder_data / "hallway.png"
rgb_image = Image.open(image_file).convert("RGB")
rgb_array = np.array(rgb_image)

# Μετατροπή της εικόνας σε γκρι κλίμακα
gray_image = color.rgb2gray(rgb_array)

# 1. Ανίχνευση ακμών με το φίλτρο Canny
edge_map = feature.canny(
    image=gray_image,
    sigma=2.0,           # αυξημένο sigma για μείωση θορύβου
    low_threshold=0.05,  # κατώφλι χαμηλής ευαισθησίας
    high_threshold=0.2   # κατώφλι υψηλής ευαισθησίας
)

# 2. Εφαρμογή πιθανοκρατικού Hough για εξαγωγή ευθειών
lines = transform.probabilistic_hough_line(
    edge_map,
    threshold=10,     # όριο
    line_length=60,   # ελάχιστο μήκος γραμμής
    line_gap=10       # μέγιστο κενό σύζευξης
)

# 3. Επικάλυψη των ανιχνευμένων ευθειών στην αρχική εικόνα
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(rgb_array)
for (start, end) in lines:
    x0, y0 = start
    x1, y1 = end
    ax.plot([x0, x1], [y0, y1], linewidth=2, color='cyan')

ax.axis("off")
ax.set_title(f"Ανιχνευμένες ευθείες Hough: {len(lines)}")

# Αποθήκευση του αποτελέσματος με ελληνικό όνομα αρχείου
output_name = "Hallway_hough.png"
fig.savefig(output_name, bbox_inches="tight")

# Εκτύπωση του ονόματος αρχείου εξόδου
print(output_name)
