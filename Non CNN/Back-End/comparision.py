from check import *

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

picture = input('Enter picture name to verify: ')

images = np.array(images, dtype='float') * 131072.0
images = list(images)

for img_path in glob.iglob('../for-verification/*.png'):
	if picture not in img_path:
		continue
	# print(img_path)
	img = Image.open(img_path)
	img = img.resize((512, 256))
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
	plt.imshow(img)
	plt.plot()
	plt.show()
	images.append(img)

images = np.array(images, dtype='float') / 131072.0
# print(images[1])
for i in range (len(images)-1):
	# print(i)
	# print(KL(images[i], images[len(images)-1]))
	# print(KL(images[len(images)-1], images[i]))
	# print(images[len(images)-1])
	if abs(KL(images[i], images[len(images)-1])) < validity_value:
		if abs(KL(images[len(images)-1], images[i])) < validity_value:
			print("Genuine")
			break
else:
	print("Fake")