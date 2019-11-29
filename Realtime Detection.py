import cv2

# Creating Function that draws rectangle and writes 'Face' indicator on the detected face


def draw_rectangle(img, classifier, scale, minNeighbour, text, color, font_family, thickness):
    faces = classifier.detectMultiScale(img, scale, minNeighbour)
    bananas = classifier.detectMultiScale(img, scale, minNeighbour)
    # x y w h are coordinates of face.
    for (x, y, w, h) in faces:
        # Drawing rectangle
        #           Starting point  End point
        #                    |         |
        cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness)
        # Writing The text
        #  Starting Point (where to start typing the text)
        #                        |
        cv2.putText(img, text, (x,y), font_family, scale, color, thickness)
        for (x, y, w, h) in bananas:
            # Drawing rectangle
            #           Starting point  End point
            #                    |         |
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            # Writing The text
            #  Starting Point (where to start typing the text)
            #                        |
            cv2.putText(img, text, (x, y), font_family, scale, color, thickness)
    return img


# Initializing Face cascade with XML file from GIT
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
bananaCascade = cv2.CascadeClassifier('BananaCascade.xml')
# Capturing the Live stream video
video_capture = cv2.VideoCapture(0)  # 0 for built-in's and -1 for external video cameras

while True:
    # reading Frames from the video
    _, img = video_capture.read()
    # Calling the method that detects face on frame and draws rectangle on it
    draw_rectangle(img, faceCascade, 1.1, 10, 'Face', (255,0,0), cv2.FONT_HERSHEY_SIMPLEX, 2)
    draw_rectangle(img, bananaCascade, 1.1, 5, 'Banana', (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX, 2)
    # Showing video process in the new window
    cv2.imshow("", img)
    # Making whether user types 'q' to quit from the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

