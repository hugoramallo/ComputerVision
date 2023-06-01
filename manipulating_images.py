#libreria para imprimir gráficas e imágenes
import matplotlib.pyplot as plt
#libreria para manejar imágenes
from PIL import Image
#libreria para manejar arrays
import numpy as np

#asignamos la imagen en formato array a una variable
baboon = np.array(Image.open('images/baboon.png'))
#establecemos tamaño en pulgadas
plt.figure(figsize=(5,5))
#preparamos la imagen para mostrarla
plt.imshow(baboon)
#mostramos la imagen
#plt.show()

#si no hacemos una copia de la imagen, las dos variables apuntarán a la misma dirección de memoria
A = baboon
#misma dirección de memoria
print(f' Direcciones de memoria iguales: Imagen A:{id(A)}, Imagen baboon:{id(baboon)}')
id(A) == id(baboon)

#hacemos una copia lo cual hace que cambie la dirección de memoria de la imagen
B = baboon.copy()
#comprobamos y da false, por tanto la dirección es diferente
print(f' Direcciones de memoria diferentes: Imagen B: {id(B)}, Imagen baboon: {id(baboon)}')
id(B)==id(baboon)
# si seteamos los valores a 0 de baboon afectará a la variable A, aparecerá en negro
baboon[:,:,] = 0

#comparamos imágenes en baboon y el array A, y efectivamente se muestran en negro,
plt.figure(figsize=(10,10))
#crea una figura con una fila y dos columnas, para posicionar las dos imágenes
plt.subplot(121)
#muestra la imagen del baboon
plt.imshow(baboon)
#título
plt.title("baboon")
#al otro lado la del array A
plt.subplot(122)
plt.imshow(A)
plt.title("array A")
#plt.show()

#comparamos baboon y el array B, baboon a 0 no afectó a la copia de la imagen del array B
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(B)
plt.title("array B")
#plt.show()

#volteamos imagen del gato
#mostramos la imagen
image = Image.open("images/cat.png")
plt.figure(figsize=(10,10))
plt.imshow(image)
#plt.show()

#hacemos un cast de la imagen a un array
array = np.array(image)
width, height, C = array.shape
print('width, height, C', width, height, C)

#giramos la imagen
array_flip = np.zeros((width, height, C), dtype=np.uint8)
for i,row in enumerate(array):
    array_flip[width - 1 - i, :, :] = row

#mostramos la imagen al revés
plt.imshow(array_flip)
#plt.show()

#usando imageOPS, volteamos también la imagen
from PIL import ImageOps
im_flip = ImageOps.flip(image)
#plt.figure(figsize=(5,5))
plt.imshow(im_flip)
#plt.show()

#en modo espejo
im_mirror = ImageOps.mirror(image)
#plt.figure(figsize=(5,5))
#plt.imshow(im_mirror)
#plt.show()

#con transpose podemos voltear la imagen con más parámetros
im_flip = image.transpose(1)
plt.imshow(im_flip)
#plt.show()
#creamos un diccionario con los valores de rotación de la imagen
flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE,
        "TRANSVERSE": Image.TRANSVERSE}
#accedemos a un tipo de rotación
flip["FLIP_LEFT_RIGHT"]

#podemos imprimir la imagen con cada rotación, delimitada por los el número de items del diccionario
#for key, values in flip.items():
    #plt.figure(figsize=(10,10))
    #plt.subplot(1,2,1)
    #plt.imshow(image)
    #plt.title("original")
    #plt.subplot(1,2,2)
    #usamos transpose
    #plt.imshow(image.transpose(values))
    #plt.title(key)
    #plt.show()

#haciendo un crop o "cortando" la imagen
upper = 150
lower = 400
crop_top = array[upper: lower,:,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_top)
plt.show()

#cortamos horizontalmente
left = 150
right = 400
crop_horizontal = crop_top[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(crop_horizontal)
plt.show()


#también se puede cropear de PIL image usando crop()
image = Image.open("images/cat.png")
crop_image = image.crop((left, upper, right, lower))
plt.figure(figsize=(5,5))
plt.imshow(crop_image)
plt.show()

#la volteamos
crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
crop_image

#usamos imagedraw
from PIL import ImageDraw
image_draw = image.copy()
image_fn = ImageDraw.Draw(im=image_draw)
shape = [left, upper, right, lower]
#podemos dibujar un rectángulo usando la función rectángulo
image_fn.rectangle(xy=shape,fill="red")
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

#podemos usar otras figuras

from PIL import ImageFont
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

image_lenna = Image.open("images/lenna.png")
array_lenna = np.array(image_lenna)

#pegamos una imagen encima de otra imagen
array_lenna[upper:lower,left:right,:]=array[upper:lower,left:right,:]
plt.imshow(array_lenna)
plt.show()

#otra superposición de la imagen
image_lenna.paste(crop_image, box=(left,upper))
plt.imshow(image_lenna)
plt.show()

#abrimos la imagen del gato y creamos otra copia
#nueva copia nueva dirección de memoria al usar el método "copy"
image = Image.open("images/cat.png")
new_image=image
copy_image=image.copy()
#comprobamos la dirección de memoria si es la misma
id(image)==id(new_image)

#otras opciones
"""
image_fn= ImageDraw.Draw(im=image)
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
image_fn.rectangle(xy=shape,fill="red")

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(new_image)
plt.subplot(122)
plt.imshow(copy_image)
plt.show().plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(new_image)
plt.subplot(122)
plt.imshow(copy_image)
plt.show()
"""

im = Image.open("images/lenna.jpg")
im_flip = ImageOps.flip(im)
plt.imshow(im_flip)
plt.show()
im_mirror = ImageOps.mirror(im)
plt.imshow(im_mirror)
plt.show()






