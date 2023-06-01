import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()
"""
#geometric transformations
image = Image.open("images/lenna.png")
plt.imshow(image)
plt.show()

#escalamos
width, height = image.size
new_width = 2 * width
new_hight = height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#escalmos el eje vertical x2
new_width = width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#escalamos ambos ejes por 2 para doblar el tamaño
new_width = 2 * width
new_hight = 2 * height
new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#reducimos la imagen
new_width = width // 2
new_hight = height // 2

new_image = image.resize((new_width, new_hight))
plt.imshow(new_image)
plt.show()

#rotamos la imagen
theta = 45
new_image = image.rotate(theta)
plt.imshow(new_image)
plt.show()
"""
#operaciones matemáticas (con arrays)
image = Image.open("images/lenna.png")
image = np.array(image)

#agregamos una constante al array de imagen
new_image = image + 20
plt.imshow(new_image)
plt.show()

#multiplicamos la intensidad de los pixeles
new_image = 10 * image
plt.imshow(new_image)
plt.show()

#generamos un array de ruido aleatorio
height = image.size
width = image.size
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
Noise.shape

#añadimos ruido a la imagen
new_image = image + Noise
plt.imshow(new_image)
plt.show()

#lo mismo pero multiplicando
new_image = image*Noise

plt.imshow(new_image)
plt.show()

#operaciones con matrices

#Grayscale images are matrices. Consider the following grayscale image:

im_gray = Image.open("barbara.png")
#Even though the image is gray, it has three channels; we can convert it to a one-channel image.

from PIL import ImageOps
im_gray = ImageOps.grayscale(im_gray)

#We can convert the PIL image to a numpy array:
im_gray = np.array(im_gray )
plt.imshow(im_gray,cmap='gray')
plt.show()

#aplicamos algoritmos diseñados para matrices

U, s, V = np.linalg.svd(im_gray , full_matrices=True)
#S no es rectangular
s.shape
#convertimos en una matriz diagonal

#imprimimos matriz U y V
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
#plot_image(U, V, title_1="Matrix U", title_2="Matrix V")
#plt.imshow(S, cmap='gray')
#plt.show()


#podemos hallar el producto de dos matrices
B = S.dot(V)
plt.imshow(B,cmap='gray')
plt.show()

A = U.dot(B)

plt.imshow(A,cmap='gray')
plt.show()

#eliminamos conlumnas de S y V
for n_component in [1,10,100,200, 500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()