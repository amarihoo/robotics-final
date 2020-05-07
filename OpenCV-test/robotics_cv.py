import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('./objects_1.png')
#plt.imshow(im)
#plt.show()

#light_orange = (1, 190, 200)
#dark_orange = (18, 255, 255)

#mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
#result = cv2.bitwise_and(nemo, nemo, mask=mask)

# Convert to grayscale and threshold
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,1,255,0)

# Find contours, draw on image and save
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)
cv2.imwrite('result1.png',im)

# Show user what we found
for cnt in contours:
   #(x,y),radius = cv2.minEnclosingSquare(cnt)
   #center = (int(x),int(y))
   #radius = int(radius)
   #print('Contour: centre {},{}, radius {}'.format(x,y,radius))
   object = cv2.minAreaRect(cnt)
   angle = object[-1]
   #print(angle)
   if angle < -45:
      angle = (90 + angle)
   # otherwise, just take the inverse of the angle to make
   # it positive
   else:
      angle = -angle
   
   print('Object angle: {}'.format(angle))
   print('Object top-left corner coordinates: {}'.format(object[0]))
#plt.subplot(1, 2, 1)
#plt.imshow(mask, cmap="gray")
#plt.subplot(1, 2, 2)
#plt.imshow(result)
#plt.show()
