# Yc 1
import cv2 
# Input image
img = cv2.imread("./assets/input.jpg")
# Show image 
cv2.imshow("Image: ", img)
# Click one button to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
# Out image conver to png
cv2.imwrite("./assets/output.png", img)
# Convert color space
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
# Convert to HSV and LAB color space
hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
# conver to lab color 
labImage = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);

cv2.imshow("Gray Image: ", grayImage)
cv2.imshow("HSV Image: ", hsvImage)
cv2.imshow("LAB Image: ", labImage)
#  out show image 
cv2.waitKey(0)
cv2.destroyAllWindows()
# Crop vu ngf toa do x: 100-400, y: 100-400
cropImage = img[100:400, 100:400]
# Resize image to 200x200
resizedImage = cv2.resize(img, (200, 200))
# Resize image with scale factor 0.5
resizedImage2 = cv2.resize(img, None, fx=0.5, fy=0.5)

cv2.imshow("Crop Image: ", cropImage)
cv2.imshow("Resized Image: ", resizedImage)
cv2.imshow("Resized Image 2: ", resizedImage2)
# out show image
cv2.waitKey(0)
cv2.destroyAllWindows()