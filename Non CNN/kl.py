import glob
import cv2
import scipy
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import warnings
set(gca,'YDir','reverse');
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

images = []
names = []
for img_path in glob.iglob('/Users/praky/Desktop/Work/SigVerification/*.png'):
	# print(img_path)
	names.append(img_path)
	img = Image.open(img_path)
	img = img.resize((512, 256))
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
	plt.imshow(img)
	plt.plot()
	plt.show()
	images.append(img)
images = np.array(images, dtype='float') / 131072.0
print(len(images))

# values1 = images[2]
# values2 = images[1]
kl_values = []
values = []
for i in range(len(names)):
	names[i] = names[i][42:],

for i in range (len(images)):
	value1 = images[i]
	for j in range (0, len(images)):
		if j == i:
			continue
		value2 = images[j]

		values.append("%.2f" % KL(value1, value2))
	kl_values.append(values)
	values = []
for i in range(len(kl_values)):
	print(names[i][0], kl_values[i])

# print(KL(values1, values2))