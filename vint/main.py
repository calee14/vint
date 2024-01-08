import cv2
from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageEnhance
import numpy as np
import datetime

def check_and_reformat(datetime_str):
    try:
        # If parsing fails, reformat the datetime string
        original_datetime = datetime.datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        datetime_str = original_datetime.strftime("%m %d '%y %H:%M:%S")
    except ValueError:
        current_timestamp = datetime.datetime.now()
        datetime_str = current_timestamp.strftime("%m %d '%y %H:%M:%S")
        
    

    return datetime_str

def blurr_scalar(width, height):
    if width >= 3000 and height >= 3000:
        return 1.8
    elif width >= 2000 and height >= 2000:
        return 1.5
    elif width >= 1000 and height >= 1000:
        return 1.2
    else:
        return 1.1
     
def rand_noise_intensity(width, height):
    if width >= 3000 and height >= 3000:
        return 17
    elif width >= 2000 and height >= 2000:
        return 10
    elif width >= 1000 and height >= 1000:
        return 7
    else:
        return 1.1
def apply_neon_effect(image, intensity=2.0, radius=3):

    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))

    enhancer = ImageEnhance.Contrast(blurred_image)
    blurred_image = enhancer.enhance(intensity)

    result_image = Image.alpha_composite(image.convert("RGBA"), blurred_image)
    
    return result_image

def add_timestamp(image, timestamp):
    # Choose font and size
    font_path = 'digital-7.ttf'  # Replace with the actual path to your font file
    # Convert the image to RGB (OpenCV uses BGR)  
    cv2_im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
    
    # Pass the image to PIL  
    
    pil_im = Image.fromarray(cv2_im_rgb)  
    pil_im = pil_im.convert('RGBA')
    
    text_mask = Image.new('RGBA', pil_im.size, (0,0,0,0))
    draw = ImageDraw.Draw(text_mask)  

    # use a truetype font  
    font_size = image.shape[1]*0.03
    font = ImageFont.truetype(font_path, font_size)  
    font_color = (216,116,40)  # white color in RGB format

    # Draw the text  
    text_position = (image.shape[1]-font_size*10, image.shape[0]*0.95)
    draw.text(text_position, timestamp, font=font, fill=font_color) 
    image = text_mask.filter(ImageFilter.BoxBlur(10))
    text_mask = apply_neon_effect(text_mask)

    pil_im.paste(text_mask, (0,0), text_mask)

    # Get back the image to OpenCV  
    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGBA2BGR)  

    return image

def eighties_digital_camera_filter(image, timestamp=None):
    '''
    image: color format bgr
    '''
    
    # color balance adjustment (warmer)
    # alpha = weight of first image, beta = weight of second image, gamma = bias (added to final image)
    image = cv2.addWeighted(image, 1.3, np.full(image.shape, np.array([0, 0, 78]), dtype=image.dtype), 0.07, -26)

    # contrast and Saturation adjustment
    image = cv2.convertScaleAbs(image, alpha=0.8, beta=30)

    # vignetting
    rows, cols, _ = image.shape
    vignette = cv2.getGaussianKernel(rows, rows*0.5) * cv2.getGaussianKernel(cols, cols*0.5).T
    # creating mask and normalising by using np.linalg function
    mask = 200 * vignette / np.linalg.norm(vignette) 
    # applying the mask to each channel in the input image
    min_mask = 0.27
    for i in range(3):
        if mask[0][0] < min_mask:
            image[:,:,i] = image[:,:,i] * (min_mask/mask[0][0])*mask
        else:
            image[:,:,i] = image[:,:,i] * mask

    
    if not timestamp:
        current_timestamp = datetime.datetime.now()
        timestamp = current_timestamp.strftime('%Y %m %d %H:%M:%S')
    else: 
        timestamp = check_and_reformat(timestamp)

    image = add_timestamp(image, timestamp)

    # add grain/noise
    intensity = rand_noise_intensity(image.shape[0], image.shape[1])
    noise = np.random.randint(-intensity, intensity + 1, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # # adjust sharpness
    image = cv2.GaussianBlur(image, (0, 0), blurr_scalar(image.shape[0], image.shape[1])) # 0.1 or 2.7 or 1.3
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

    return image
 
def nineties_digital_camera_filter(image, timestamp=None):
    '''
    image: color format bgr
    '''
    
    # color balance adjustment (warmer)
    # alpha = weight of first image, beta = weight of second image, gamma = bias (added to final image)
    image = cv2.addWeighted(image, 1.13, np.full(image.shape, np.array([0, 10, 10]), dtype=image.dtype), 0.07, -13)

    # contrast and brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=1.07, beta=0.8)
    
    # saturation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_factor = 0.98  # Increase or decrease saturation (1.0 means no change)
    hsv_image[:, :, 1] = np.clip(saturation_factor * hsv_image[:, :, 1], 0, 170).astype(np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # vignetting
    rows, cols, _ = image.shape
    vignette = cv2.getGaussianKernel(rows, rows*0.5) * cv2.getGaussianKernel(cols, cols*0.5).T
    # creating mask and normalising by using np.linalg function
    mask = 255 * vignette / np.linalg.norm(vignette) 
    # applying the mask to each channel in the input image
    min_mask = 0.33
    for i in range(3):
        if mask[0][0] < min_mask:
            image[:,:,i] = image[:,:,i] * (min_mask/mask[0][0])*mask
        else:
            image[:,:,i] = image[:,:,i] * mask

    
    if not timestamp:
        current_timestamp = datetime.datetime.now()
        timestamp = current_timestamp.strftime('%Y %m %d %H:%M:%S')
    else: 
        timestamp = check_and_reformat(timestamp)

    image = add_timestamp(image, timestamp)

    # add grain/noise
    intensity = rand_noise_intensity(image.shape[0], image.shape[1])
    noise = np.random.randint(-intensity, intensity + 1, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # # adjust sharpness
    image = cv2.GaussianBlur(image, (0, 0), blurr_scalar(image.shape[0], image.shape[1])) # 0.1 or 2.7 or 1.3
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

    return image

def aesthetic_digital_camera_filter(image, timestamp=None):

    # color balance adjustment (warmer)
    # alpha = weight of first image, beta = weight of second image, gamma = bias (added to final image)
    # image = cv2.addWeighted(image, 1.13, np.full(image.shape, np.array([0, 0, 65]), dtype=image.dtype), 0.07, -13)

    # contrast and brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=0.91, beta=0.2)
    
    # saturation
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # saturation_factor = 0.91  # Increase or decrease saturation (1.0 means no change)
    # hsv_image[:, :, 1] = np.clip(saturation_factor * hsv_image[:, :, 1], 0, 180).astype(np.uint8)
    # image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # vignetting
    rows, cols, _ = image.shape
    vignette = cv2.getGaussianKernel(rows, rows*0.5) * cv2.getGaussianKernel(cols, cols*0.5).T
    # creating mask and normalising by using np.linalg function
    mask = 255 * vignette / np.linalg.norm(vignette) 
    # applying the mask to each channel in the input image
    min_mask = 0.33
    for i in range(3):
        if mask[0][0] < min_mask:
            image[:,:,i] = image[:,:,i] * (min_mask/mask[0][0])*mask
        else:
            image[:,:,i] = image[:,:,i] * mask

    
    if not timestamp:
        current_timestamp = datetime.datetime.now()
        timestamp = current_timestamp.strftime('%Y %m %d %H:%M:%S')
    else: 
        timestamp = check_and_reformat(timestamp)

    image = add_timestamp(image, timestamp)

    # add grain/noise
    intensity = rand_noise_intensity(image.shape[0], image.shape[1])
    noise = np.random.randint(-intensity, intensity + 1, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # # adjust sharpness
    image = cv2.GaussianBlur(image, (0, 0), blurr_scalar(image.shape[0], image.shape[1])) # 0.1 or 2.7 or 1.3
    image = cv2.addWeighted(image, 1.2, image, 0.1, 0)


    return image

image_file = 'caye1.jpeg'
# get the timestamp from the image metadata
timestamp = 'hi' #Image.open(image_file)._getexif()[36867]
image = cv2.imread(image_file)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = aesthetic_digital_camera_filter(image, timestamp)

# display the result
cv2.imwrite(image_file[:-5]+'digital'+'.jpeg', image)
cv2.imshow('Vintage Effect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
