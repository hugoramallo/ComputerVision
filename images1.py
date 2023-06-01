from PIL import Image
import matplotlib.pyplot as plt
#importamos la libreria del SO
import os
#nombre de la imagen
my_image = "lenna.png"
#devuelve al directorio actual, en este caso el de python
cwd = os.getcwd()
#Guardamos la ruta de la imagen y su nombre
image_path = os.path.join(cwd, my_image)
#abrimos la imagen
image = Image.open(my_image)
image = Image.open(my_image)
#comprobamos que realmente es una imagen
print(type(image))
#mostramos el tamaño de la imagen
print (image.size)
#ver el pixel format RGB
print (image.mode)

#comprobar intensidad de la imagen
#im = image.load()
#x = 0
#y = 1
#im[y,x]

#guardamos la imagen en otro formato
image.save("lenna.jpg")

#escala de grises
from PIL import ImageOps
image_gray = ImageOps.grayscale(image)
#Quantization 256 representa el rango de grises y // redondea el resultado a un entero, dando 128
image_gray.quantize(256 // 2)
#mostramos la imagen en escala de grises
#image_gray.show()



#mostramos la imagen en pantalla
#image.show()
#lo mostramos con plot para verla dentro de un eje X e Y
#plt.figure(figsize=(10,10))
#plt.imshow(image_gray)
#plt.show()

#función que concatena dos imágenes lado a lado
def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


#método que compara dos imágenes en escala de grises, en diferentes rangos de la escala de 0 a 255
#get_concat_h(image_gray,  image_gray.quantize(256//2)).show(title="Lena")
#for n in range(3,8):
#    plt.figure(figsize=(10,10))
#    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n)))
#    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
#    plt.show()

#Trabajamos con otros canales de color
baboon = Image.open('images/baboon.png')
#obtenemos los colores RGB separados y se las asignamos a las variables
red, green, blue = baboon.split()
#obtenemos el rojo
get_concat_h(baboon, red)
#obtenemos el azul
get_concat_h(baboon, blue)
#obtenemos el verde
get_concat_h(baboon, green)

#PIL IMAGES INTO NUMPY ARRAYS

import numpy as np

#convertimos la imagen en un array
array= np.asarray(image)
print(type(array))

# devuelve una tupla con el tamaño del array
print(array.shape)
#imprimimos el array con los datos de la imagen
print(array)

#intensidad de los valores a 8 bits
array[0, 0]

#podemos averiguar la intensidad máxima o mínima para el array
array.min()
array.max()

#Imprimimos el array como una imagen
#figsize especifica el ancho y alto de la imagen en pulgadas
plt.figure(figsize=(10,10))
plt.imshow(array)
plt.show()

#podemos especificar las filas que queremos en el array
#como la imagen es de 512, mostraria la mitad
rows = 256
#imprimimos
plt.figure(figsize=(10,10))
plt.imshow(array[0:rows,:,:])
plt.show()

#lo mismo con las columnas
columns = 256
plt.figure(figsize=(10,10))
plt.imshow(array[:,0:columns,:])
plt.show()

#podemos reasignar el array a otra variable
A = array.copy()
plt.imshow(A)
plt.show()

#podemos trabajar con diferentes canales de color
baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array)
plt.show()

#cambiamos la intensidad en el canal rojo
baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array[:,:,0], cmap='gray')
plt.show()

#creamos un array y seteamos los colores a 0 baboon_red=baboon_array.copy()
baboon_red=baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.show()

#lo mismo en azul
baboon_blue=baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()

#lo hacemos con la imagen de lena
lenna = Image.open('images/lenna.jpg')
#convertimos la imagen de lenna a un array
lena_array = np.array(lenna)
#asignamos a la variable una copia del array, para quitarle el canal
blue_lenna = lena_array.copy()
blue_lenna[:,:,0] = 0
blue_lenna[:,:,1] = 0
#10 x 10 inches
plt.figure(figsize=(10,10))
#preparamos la matrix para mostrar
plt.imshow(blue_lenna)
#show the image
plt.show()


image.transpose(my_image.FLIP_TOP_BOTTOM)
plt.imshow(my_image)
plt.show()


