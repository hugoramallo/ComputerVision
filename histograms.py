import matplotlib.pyplot as plt
import cv2
import numpy as np

#plot two images side by side
def plot_image(image_1, image_2,title_1="Orignal", title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

#plot two histograms side by side
def plot_hist(old_image, new_image,title_old="Orignal", title_new="New Image"):
    intensity_values=np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()
"""
#imprimos un array de color negro, blanco y grises
toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]],dtype=np.uint8)
plt.imshow(toy_image, cmap="gray")
plt.show()
print("toy_image:",toy_image)

#histograma de barras
plt.bar([x for x in range(6)],[1,5,2,0,0,0])
plt.show()

plt.bar([x for x in range(6)],[0,1,0,5,0,2])
plt.show()

#histograma en escala de grises
goldhill = cv2.imread("images/goldhill.bmp",cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,10))
plt.imshow(goldhill,cmap="gray")
plt.show()

#calcula el histograma
hist = cv2.calcHist([goldhill],[0], None, [256], [0,256])

#imprimimos un gráfico de barras
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:,0], width = 5)
plt.title("Bar histogram")
plt.show()

#probability mass function / normalize
PMF = hist / (goldhill.shape[0] * goldhill.shape[1])

#imprimir como una función continua
plt.plot(intensity_values,hist)
plt.title("histogram")
plt.show()

#podemos aplicar el histograma a cada canal de color de la imagen RGB
baboon = cv2.imread("images/baboon.png")
plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
plt.show()

#imprime el histograma de la imagen con los valores de los colores que predominan en ella
color = ('blue', 'green', 'red')
for i, col in enumerate(color):
    histr = cv2.calcHist([baboon], [i], None, [256], [0, 256])
    plt.plot(intensity_values, histr, color=col, label=col + " channel")

    plt.xlim([0, 256])
plt.legend()
plt.title("Histogram Channels")
plt.show()

#imágenes negativas
neg_toy_image = -1 * toy_image + 255

print("toy image\n", neg_toy_image)
print("image negatives\n", neg_toy_image)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(toy_image,cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(neg_toy_image,cmap="gray")
plt.show()
print("toy_image:",toy_image)
"""
"""
#cambiar la intensidad de los pixeles, brillo, oscuridad, etc puede ser muy útil
#este ejemplo es una mamografía
image = cv2.imread("images/mammogram.png", cv2.IMREAD_GRAYSCALE)
cv2.rectangle(image, pt1=(160, 212), pt2=(250, 289), color = (255), thickness=2)

plt.figure(figsize = (10,10))
plt.imshow(image, cmap="gray")
plt.show()

#aplicamos la transformación de intensidad
img_neg = -1 * image + 255
#podemos ver micro-calcificaciones en imágenes negativas (invirtiendo colores)
plt.figure(figsize=(10,10))
plt.imshow(img_neg, cmap = "gray")
plt.show()
"""
"""
#ajustes de contraste y brillo
goldhill = cv2.imread("images/goldhill.bmp",cv2.IMREAD_GRAYSCALE)
alpha = 1 # Simple contrast control
beta = 100   # Simple brightness control
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
#imprimos la imagen más brillante
plot_image(goldhill, new_image, title_1 = "Orignal", title_2 = "brightness control")

#incrementamos el contraste
plt.figure(figsize=(10,5))
alpha = 2# Simple contrast control
beta = 0 # Simple brightness control   # Simple brightness control
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)

#imprimimos la imagen con su correspondiente histograma
plot_image(goldhill,new_image,"Orignal","contrast control")

plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image,"Orignal","contrast control")

#al imprimir la imagen vemos que es muy brillante, podemos adaptar el brillo al hacer la imagen más oscura
plt.figure(figsize=(10,5))
alpha = 3 # Simple contrast control
beta = -200  # Simple brightness control
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)

plot_image(goldhill, new_image, "Orignal", "brightness & contrast control")

plt.figure(figsize=(10,5))
plot_hist(goldhill, new_image, "Orignal", "brightness & contrast control")

#Equualización del histograma

zelda = cv2.imread("images/zelda.png",cv2.IMREAD_GRAYSCALE)
new_image = cv2.equalizeHist(zelda)
plot_image(zelda,new_image,"Orignal","Histogram Equalization")
plt.figure(figsize=(10,5))
plot_hist(zelda, new_image,"Orignal","Histogram Equalization")
"""
#Thresholding and simple segmentation
def thresholding(input_img, threshold, max_value=255, min_value=0):
    N, M = input_img.shape
    image_out = np.zeros((N, M), dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            if input_img[i, j] > threshold:
                image_out[i, j] = max_value
            else:
                image_out[i, j] = min_value
    return image_out

#toy image creada con un array
toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]],dtype=np.uint8)

#aplicamos thresholding (sirve para convertir imagen en blanco y negro o resltar areas de interes basadas en su color, intesndid)
threshold = 1
max_value = 2
min_value = 0
thresholding_toy = thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
thresholding_toy

#comparamos las imágenes
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(toy_image, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(thresholding_toy, cmap="gray")
plt.title("Image After Thresholding")
plt.show()

#cameraman imagen
image = cv2.imread("images/cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.show()
#histograma con dos picos
goldhill = cv2.imread("images/goldhill.bmp",cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([goldhill], [0], None, [256], [0, 256])
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:, 0], width=5)
plt.title("Bar histogram")
plt.show()

#cameraman corresponde los pixeles más oscuros
threshold = 87
max_value = 255
min_value = 0
new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)
plot_image(image, new_image, "Orignal", "Image After Thresholding")
plt.figure(figsize=(10,5))
plot_hist(image, new_image, "Orignal", "Image After Thresholding")

#with threshold
cv2.THRESH_BINARY
ret, new_image = cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")

#with threshold, will not change the values if the pixels are less than the threshold value:
ret, new_image = cv2.threshold(image,86,255,cv2.THRESH_TRUNC)
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")

ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
plot_image(image,otsu,"Orignal","Otsu")
plot_hist(image, otsu,"Orignal"," Otsu's method")