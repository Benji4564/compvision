import cv2
import numpy as np

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("D:/Dokumente/Git/compvision/haarcascade.xml")

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Load the first face to compare
face1 = cv2.imread("images/b.jpg", 0)
face1 = cv2.resize(face1, (64, 64))

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face2 = gray[y:y+h, x:x+w]

        # Resize the face to the standard size
        face2 = cv2.resize(face2, (64, 64))

        # Compute the mean squared error between the two faces
        mse = np.mean((face1 - face2) ** 2)
        print(mse)

        # If the MSE is below a certain threshold, conclude that the faces are the same
        if mse > 100:
            text = "Same Person"
            color = (0, 255, 0)
        else:
            text = "Different Person"
            color = (0, 0, 255)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put the text on the frame
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow("Face Comparison", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
camera.release()

# Close all windows
cv2.destroyAllWindows()
