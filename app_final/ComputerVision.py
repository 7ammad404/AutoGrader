import cv2
import matplotlib.pyplot as plt
import numpy as np

def computer_vision_soft_version(file):
    """
    This function takes a AG test photo and returns the student answer box

    Parameters
    file :  the Photo of the AG test in opencv file object note high quality as possible
    """
    
    img = file
    imgr = img.copy()
    # convert it to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # binarzation
    img_bin = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)[1]
    # invert the image
    img_bininv = cv2.bitwise_not(img_bin)
    # Apply morphological operations to remove small noise regions Note use this if the image low quality only
    # Note use this if the image low quality only and if there is a error on the number of answers otherwise leave commented out
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #img_bininv = cv2.morphologyEx(img_bininv, cv2.MORPH_OPEN, kernel)

    # getting the contours
    contours , hier = cv2.findContours(img_bininv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # now get each answer in the photo
    Answers = []
    for contour in contours:
        # getting the answer dimensions 
        x, y, w, h = cv2.boundingRect(contour)
        # cropping based on the dimensions
        cropped_image = imgr[y:y+h, x:x+w]
        Answers.append(cropped_image)
    height, width, channels = imgr.shape
    return Answers , contours , height, width

def computer_vision_scanned_version(img , contours, height, width):
    img_scanned_resized = cv2.resize(img, (width, height))
    Answers = []
    for contour in contours:
        # getting the answer dimensions 
        x, y, w, h = cv2.boundingRect(contour)
        # cropping based on the dimensions
        cropped_image = img_scanned_resized[y:y+h, x:x+w]
        Answers.append(cropped_image)
    return cropped_image