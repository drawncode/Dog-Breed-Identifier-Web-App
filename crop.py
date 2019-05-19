import cv2

def crop_image(path):
	desired_size = 224
	# # im_pth = "/home/jdhao/test.jpg"

	# im = cv2.imread(path)
	# old_size = im.shape[:2] # old_size is in (height, width) format

	# ratio = float(desired_size)/max(old_size)
	# new_size = tuple([int(x*ratio) for x in old_size])
	# # print(ratio,new_size,old_size)
	# # new_size should be in (width, height) format

	# im = cv2.resize(im, (new_size[1], new_size[0]))

	# delta_w = desired_size - new_size[1]
	# delta_h = desired_size - new_size[0]
	# top, bottom = delta_h//2, delta_h-(delta_h//2)
	# left, right = delta_w//2, delta_w-(delta_w//2)

	# color = [0, 0, 0]
	# new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
	#     value=color)

	# cv2.imshow("image", new_im)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	img = cv2.imread(path)
	f = open("output/details.txt","r")
	for x in f.readlines():
		x = x.split(',')
		name = x[0]
		dog = img[int(x[3]):int(x[4]),int(x[1]):int(x[2]),:]
		old_size = dog.shape[:2] # old_size is in (height, width) format

		ratio = float(desired_size)/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])
		# print(ratio,new_size,old_size)
		# new_size should be in (width, height) format

		dog = cv2.resize(dog, (new_size[1], new_size[0]))

		delta_w = desired_size - new_size[1]
		delta_h = desired_size - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)

		color = [0, 0, 0]
		new_im = cv2.copyMakeBorder(dog, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
		# dog = cv2.resize(dog,(224,224))
		cv2.imwrite("output/"+name+".jpg",new_im)