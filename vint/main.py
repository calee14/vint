import cv2
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import datetime

def check_and_reformat(datetime_str):
    try:
        # If parsing fails, reformat the datetime string
        original_datetime = datetime.datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        datetime_str = original_datetime.strftime('%Y %m %d %H:%M:%S')
    except ValueError:
        current_timestamp = datetime.datetime.now()
        datetime_str = current_timestamp.strftime('%Y %m %d %H:%M:%S')
    print(datetime_str)
    return datetime_str

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

    pil_im.paste(text_mask, (0,0), text_mask)

    # Get back the image to OpenCV  
    image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGBA2BGR)  

    return image
   
def digital_camera_filter(image, timestamp=None):
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
    intensity = 5
    noise = np.random.randint(-intensity, intensity + 1, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # # adjust sharpness
    image = cv2.GaussianBlur(image, (0, 0), 1.3) # 0.1 or 2.7 or 1.3
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

    return image

image_file = 'jess2.jpeg'
# get the timestamp from the image metadata
timestamp = Image.open(image_file)._getexif()[36867]
image = cv2.imread(image_file)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = digital_camera_filter(image, timestamp)

# display the result
cv2.imwrite(image_file[:-5]+'digital'+'.jpeg', image)
cv2.imshow('Vintage Effect', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


