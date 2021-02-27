import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
path= cv2.imread('/home/nico/Documents/MI204/TP1/TP1_Features/Image_Pairs/FlowerGarden2.png',0)
print('pas de soucis0')
img=np.float64(path)
print('pas de soucis1')
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    #dx = -2*img[y, x+1] + 2*img[y, x-1] + img[y-1, x-1] - img[y-1, x+1] + img[y+1, x-1] - img[y+1, x+1]   #derivate x
    #dy = 2*img[y+1, x] - 2*img[y-1, x] - img[y-1, x-1] + img[y+1, x+1] - img[y-1, x+1] + img[y+1, x-1]   #derivate y
    dx = 2*img[y-1, x] - 2*img[y+1, x] - img[y+1, x-1] + img[y-1, x-1] - img[y+1, x+1] + img[y-1, x+1]   #derivate x correctas
    dy = 2*img[y, x+1] - 2*img[y, x-1] - img[y+1, x-1] + img[y+1, x+1] - img[y-1, x-1] + img[y-1, x+1]   #derivate y Correctas
    ngrad=(dx**2+dy**2)**(1/2)
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1]
#    img2[y,x] = min(max(val,0),255)
    img2[y,x]=ngrad

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")


plt.subplot(121)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
ksobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
ksobel_y = np.array([[-1 , -2, -1],[0, 0, 0],[1, 2, 1]])

dx2=cv2.filter2D(img,-1,ksobel_x)
dy2=cv2.filter2D(img,-1,ksobel_y)
ngrad2=(dx2**2+dy2**2)**(1/2)
img3=ngrad2
#img3 = cv2.filter2D(img,-1,kernel)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

plt.show()
