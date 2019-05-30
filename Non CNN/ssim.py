# import the necessary packages
from skimage import measure
# from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from PIL import Image, ImageFilter, ImageDraw

images = []
names = []


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()



for img_path in glob.iglob('/Users/praky/Desktop/*.png'):
	print(img_path)
	if img_path[21:] == 'Genuine_1.png':
		crop(img_path, (161, 166, 706, 1050), 'cropped.jpg')
	names.append(img_path)
	img = Image.open(img_path)
	img = img.resize((903, 354))
	# plt.imshow(img)
	# plt.plot()
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	images.append(img)
# images = np.array(images, dtype='float') / 319662.0
print(len(images))


ssim_values = []
values = []

for i in range (len(images)):
	value1 = images[i]
	for j in range (0, len(images)):
		if j == i:
			continue
		value2 = images[j]

		values.append("%.2f" % measure.compare_ssim(value1, value2))
	ssim_values.append(values)
	values = []
for i in range(len(ssim_values)):
	print(names[i][0], ssim_values[i])
# s = measure.compare_ssim(imageA, imageB)