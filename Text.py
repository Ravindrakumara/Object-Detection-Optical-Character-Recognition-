# Import required packages
import cv2 as cv
import pytesseract

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Read image from which text needs to be extracted
img = cv.imread("src/sample.png")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img))
#

hImg,wImg,_ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    # print(b)
    b = b.split(' ')
    print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),2)
    cv.putText(img,b[0],(x,hImg-y),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0))

# print(img)
cv.imshow('Text Detection ',img)
cv.waitKey(0)

