import cv2
import numpy as np

# Load the haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('D:/Dokumente/Git/compvision/haarcascade.xml')

# Load the haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('D:/Dokumente/Git/compvision/haarcascade_eye.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over the faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the region of interest for the eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Iterate over the eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close the window
cv2.destroyAllWindows()
