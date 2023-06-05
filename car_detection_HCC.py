#______Car Detection with Haar Cascade Classifier______

### install opencv version 3.4.2 for this exercise,
### if you have a different version of OpenCV please switch to the 3.4.2 version
# !{sys.executable} -m pip install opencv-python==3.4.2.16
import urllib.request
import cv2

import inline as inline

#print(cv2.__version__)
from matplotlib import pyplot as plt
#%matplotlib inline

#Create a function that cleans up and displays the image
def plt_show(image, title="", gray=False, size=(12, 10)):
    from pylab import rcParams
    temp = image

    # convert to grayscale images
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)


    # change image size
    rcParams['figure.figsize'] = [10, 10]
    # remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()

#Create a function to detect cars in an image
def detect_obj(image):
    #clean your image
    plt_show(image)
    ## detect the car in the image
    object_list = detector.detectMultiScale(image)
    print(object_list)
    #for each car, draw a rectangle around it
    for obj in object_list:
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2) #line thickness
    ## lets view the image
    plt_show(image)


#Load image
## read the url
#haarcascade_url = 'car/cars.xml'
haar_name = "car/cars.xml"
#urllib.request.urlretrieve(haarcascade_url, haar_name)

#Get the detector using the `cv2.CascadeClassifier()` module on the pretrained dataset
detector = cv2.CascadeClassifier(haar_name)


## we will read in a sample image
image_url = "car/car-road-behind.jpg"
image_name = "car-road-behind.jpg"
urllib.request.urlretrieve(image_url, image_name)
image = cv2.imread(image_name)

#plot the image
plt_show(image)

#Run the function on loaded image
detect_obj(image)
()
## replace "your_uploaded_file" with your file name
my_image = cv2.imread("car.jpeg")