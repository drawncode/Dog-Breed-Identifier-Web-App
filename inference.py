import os
from crop import crop_image
from sys import argv
path = "test.jpg"
os.system("rm output/* && touch output/details.txt")
os.system("./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights "+path)
crop_image(path)
os.system("python3 code_v1.py")
from glob import glob
os.system("mv predictions.jpg output/detections.jpg && cp static/photo/dogs_result.jpg output/details.jpg")
