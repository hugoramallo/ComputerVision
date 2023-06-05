import torchvision
from torchvision import  transforms
import torch
from torch import no_grad
import requests

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#This function will assign a string name to a predicted class and eliminate predictions whose likelihood is under a threshold.
def get_predictions(pred, threshold=0.8, objects=None):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold

    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object
    thre
    """

    predicted_classes = [(COCO_INSTANCE_CATEGORY_NAMES[i], p, [(box[0], box[1]), (box[2], box[3])]) for i, p, box in
                         zip(list(pred[0]['labels'].numpy()), pred[0]['scores'].detach().numpy(),
                             list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes = [stuff for stuff in predicted_classes if stuff[1] > threshold]

    if objects and predicted_classes:
        predicted_classes = [(name, p, box) for name, p, box in predicted_classes if name in objects]
    return predicted_classes

#Draws box around each object

def draw_box(predicted_classes, image, rect_th=10, text_size=3, text_th=3):
    """
    draws box around each object

    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object
    image : frozen surface

    """

    img = (np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0,
                   1) * 255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
        label = predicted_class[0]
        probability = predicted_class[1]
        box = predicted_class[2]

        cv2.rectangle(img, box[0], box[1], (0, 255, 0), rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, label, box[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        cv2.putText(img, label + ": " + str(round(probability, 2)), box[0], cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    del (img)
    del (image)

#this function will speed up your code by freeing memory.

def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)

#load pre-trained faster R-CNN

model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

for name, param in model_.named_parameters():
    param.requires_grad = False
print("done")

#the function calls Faster R-CNN <code> model_ </code> but save RAM:
def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat

#Here are the 91 classes.

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
len(COCO_INSTANCE_CATEGORY_NAMES)

#Object Localization
#In Object Localization we locate the presence of objects in an image and indicate the location with a bounding box. Consider the image of Geoffrey Hinton

img_path='jeff_hinton.png'
half = 0.5
image = Image.open(img_path)

image.resize( [int(half * s) for s in image.size] )

plt.imshow(image)
plt.show()

#We will create a transform object to convert the image to a tensor.
transform = transforms.Compose([transforms.ToTensor()])

#We convert the image to a tensor.

img = transform(image)

#make a prediction

pred = model([img])

#35 class predictions

pred[0]['labels']

#likelihood of each class
pred[0]['scores']

#The class number corresponds to the index of the list with the corresponding  category name
index=pred[0]['labels'][0].item()
COCO_INSTANCE_CATEGORY_NAMES[index]

#we have the coordinates of the bounding box
bounding_box=pred[0]['boxes'][0].tolist()
bounding_box

#<p>top (t),left (l),bottom(b),right (r)</p>
t,l,r,b=[round(x) for x in bounding_box]

#We convert the tensor to an open CV array and plot an image with the box:

img_plot=(np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8)
cv2.rectangle(img_plot,(t,l),(r,b),(0, 255, 0), 10) # Draw Rectangle with the coordinates
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.show()
del img_plot, t, l, r, b

#We can localize objects; we do this using the function

pred_class=get_predictions(pred,objects="person")
draw_box(pred_class, img)
del pred_class

#We can set a threshold

get_predictions(pred,threshold=1,objects="person")

#Here we have no output as the likelihood is not 100%
pred_thresh=get_predictions(pred,threshold=0.98,objects="person")
draw_box(pred_thresh,img)
del pred_thresh

#delete objects to save memory

save_RAM(image_=True)

#we can locate multiple objects
img_path='DLguys.jpeg'
image = Image.open(img_path)
image.resize([int(half * s) for s in image.size])
plt.imshow(np.array(image))
plt.show()

#we can set a threshold to detect the object, 0.9 seems to work.
img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.8,)
draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

#Or we can use objects parameter:
pred_obj=get_predictions(pred,objects="person")
draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_obj

#If we set the threshold too low, we will detect objects that are not there.
pred_thresh=get_predictions(pred,threshold=0.01)
draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

#the following lines will speed up your code by using less RAM.

save_RAM(image_=True)

#object detection
img_path='istockphoto-187786732-612x612.jpeg'
image = Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
plt.imshow(np.array(image))
plt.show()
del img_path

#If we set a threshold, we can detect all objects whose likelihood is above that threshold.

img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.97)
draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
del pred_thresh

#save ram for speed

save_RAM(image_=True)

#We can specify the objects we would like to classify, for example, cats and dogs:
img_path='istockphoto-187786732-612x612.jpeg'
image = Image.open(img_path)
img = transform(image)
pred = model([img])
pred_obj=get_predictions(pred,objects=["dog","cat"])
draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_obj

# save_RAM()

#here, we set the threshold to 0.7, and we incorrectly  detect a cat

# img = transform(image)
# pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.70,objects=["dog","cat"])
draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
del pred_thresh

save_RAM(image_=True)

#detect other objects
img_path='watts_photos2758112663727581126637_b5d4d192d4_b.jpeg'
image = Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
plt.imshow(np.array(image))
plt.show()
del img_path

img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.997)
draw_box(pred_thresh,img)
del pred_thresh

save_RAM(image_=True)

#TEST MODEL WITH AN UPLOADED IMAGE

url='https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'

#request download image
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
del url

img = transform(image )
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.95)
draw_box(pred_thresh, img)
del pred_thresh

save_RAM(image_=True)

# img_path='Replace with the name of your image as seen in your directory'
# image = Image.open(img_path) # Load the image
# plt.imshow(np.array(image ))
# plt.show()

# img = transform(image )
# pred = model(img.unsqueeze(0))
# pred_thresh=get_predictions(pred,threshold=0.95)
# draw_box(pred_thresh,img)