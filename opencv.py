import cv2

# Load the classifiers downloaded
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the image and convert to grayscale format
img = cv2.imread('people3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate coordinates
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in faces:
    #         starting point     End Point    RGB     Thickness
    #                    |          |         |       |
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
    #                    text       position          FONT             Scale   RGB       Thickness
    cv2.putText(img, 'Human face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)


img = cv2.resize(img, (438, 584))
# Show the image
cv2.imshow('image', img)
cv2.waitKey(0)