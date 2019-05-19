import matplotlib.pyplot as plt
import cv2
import math

result_images=['output/dog_1.jpg' ,'output/dog_2.jpg', 'output/dog_3.jpg'] 
result_breeds=['gaurav','vedant', 'rakshit']

l=len(result_images)
# l=1
fig=plt.figure(figsize=(5,5))
if l>2:
	columns = 2
else:
	columns=l
rows=math.ceil(l/columns)
# fig, ax = plt.subplots()
# [axi.set_axis_off() for axi in ax.ravel()]
for i in range(0, l):
	print(result_images[i])
	img = cv2.imread(result_images[i])
	# print(img.shape)
	fig.add_subplot(rows, columns, i+1)
	plt.title(result_breeds[i])
	plt.imshow(img)
	cur_axes = plt.gca()
	cur_axes.axes.get_xaxis().set_visible(False)
	cur_axes.axes.get_yaxis().set_visible(False)
plt.savefig("dogs_result.jpg")
img = cv2.imread("dogs_result.jpg")
f_img = np.zeros((1000))