import matplotlib.pyplot as plt
import cv2
import numpy as np

#copiando imagenes y reasignando un array a otra variable
#cv2 toma el archivo y lo devuelve en un numpy array
baboon = cv2.imread("baboon.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

#si no aplicamos copy la nueva variable estará en la misma dirección de memoria
A = baboon

#comprobamos las direcciones de memoria, son las mismas
print(id(A)==id(baboon))
print(id(A))

#aplicamos método copy y la dirección de memoria es diferente para la copia
B = baboon.copy()
print(id(B)==id(baboon))

#ponemos baboon a 0 y todos los objetos de imagen en que estén en su dirección de memoria
#cuando los efectos de una variable afectan a la otra por estar en la misma dirección de memoria se llama aliasing
#imprimimos la comparación de las 2 imágenes y vemos que las dos están a 0
baboon[:,:,] = 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.title("array A")
plt.show()

#comparamos la imagen de B(nueva dirección de memoria) con la imagen baboon que está a 0
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
plt.title("array B")
plt.show()


#volteando imágenes
image = cv2.imread("cat.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

#hacemos un cast
width, height,C=image.shape
print('width, height,C',width, height,C)

#rotar la imagen verticalmente
array_flip = np.zeros((width, height,C),dtype=np.uint8)
#asignamos la primera fila de pixeles del array original a la última fila
for i,row in enumerate(image):
    #dibujamos al revés el array del pixeles
        array_flip[width-1-i,:,:]=row

#imprimimos el gato invertido
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(array_flip, cv2.COLOR_BGR2RGB))
plt.show()


#OpenCV tiene varias formas de girar la imagen
for flipcode in [0,1,-1]:
    #giramos la imagen
    im_flip =  cv2.flip(image,flipcode )
    plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
    plt.title("flipcode: "+str(flipcode))
    plt.show()

#rotamos con rotate()
im_flip = cv2.rotate(image,0)
plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
plt.show()

#OpenCV tiene atributos built-in para hacer diversas funciones
flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,"ROTATE_180":cv2.ROTATE_180}
flip["ROTATE_90_CLOCKWISE"]
#imprimimos usando los diferentes atributos que tenemos en el diccionario flip
for key, value in flip.items():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(cv2.rotate(image,value), cv2.COLOR_BGR2RGB))
    plt.title(key)
    plt.show()

#cropping image (cortando la imagen)
upper = 150
lower = 400
crop_top = image[upper: lower,:,:]
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
plt.show()

#cortando horizontalmente
left = 150
right = 400
crop_horizontal = crop_top[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
plt.show()

#cambiando píxeles específicos en la imagen
array_sq = np.copy(image)
array_sq[upper:lower,left:right,:] = 0

#comparamos los resultados con la nueva imagen
plt.figure(figsize=(10,10))
# Crea una figura con 1 fila y 2 columnas, selecciona la primera subtrama
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(array_sq,cv2.COLOR_BGR2RGB))
plt.title("Altered Image")
plt.show()

#Creamos figuras con OpenCV
start_point, end_point = (left, upper),(right, lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3)
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.show()

#Overlay o superponer texto en imágenes, usando la función putText
image_draw=cv2.putText(img=image,text='Stuff',org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
plt.show()

#question 4
im = cv2.imread("lenna.jpg")
#flipping
im_flip =  cv2.flip(im,0)
#mirroring
im_mirror =  cv2.flip(im, 1)
#size in inches
plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(im_mirror, cv2.COLOR_BGR2RGB))
plt.show()
