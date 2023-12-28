import cv2
import numpy as np

image = cv2.imread('mu.jpeg')
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# color balance adjustment
image = cv2.addWeighted(image, 1.2, np.zeros(image.shape, image.dtype), 0.1, -26)

# contrast and Saturation adjustment
image = cv2.convertScaleAbs(image, alpha=0.8, beta=30)

# vignetting
rows, cols, _ = image.shape
vignette = cv2.getGaussianKernel(rows, rows*0.5) * cv2.getGaussianKernel(cols, cols*0.5).T
# creating mask and normalising by using np.linalg function
mask = 1000 * vignette / np.linalg.norm(vignette) 
# applying the mask to each channel in the input image
for i in range(3):
    image[:,:,i] = image[:,:,i] * mask


# add grain/noise
intensity = 7
noise = np.random.randint(-intensity, intensity + 1, image.shape)
image = np.clip(image + noise, 0, 255).astype(np.uint8)

# # lower resolution
# image = cv2.resize(noisy_image, (cols // 2, rows // 2), interpolation=cv2.INTER_LINEAR)

# # adjust sharpness
# image = cv2.GaussianBlur(image, (0, 0), 3)
# image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

# # simulate light leaks
# cv2.line(image, (0, 0), (cols, rows), (50, 50, 50), 15)

# display the result
cv2.imshow('Vintage Effect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
