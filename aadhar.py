import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# img = cv2.imread('aadhar.jpg')
img = cv2.imread('Aadhar-card.jpg')

def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text


def get_grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,5)

def thersholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img = get_grayscale(img)
img = thersholding(img)
img = remove_noise(img)



print(ocr_core(img))
# print(cv2.__version__)
