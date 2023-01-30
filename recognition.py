# import cv2
# import numpy as np

# # Load the cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier("haarcascade.xml")

# # Initialize the webcam
# camera = cv2.VideoCapture(0)

# # Load the first face to compare
# face1 = cv2.imread("images/b.jpg", 0)
# face1 = cv2.resize(face1, (64, 64))

# while True:
#     # Read a frame from the camera
#     ret, frame = camera.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # Loop over the faces
#     for (x, y, w, h) in faces:
#         # Crop the face from the frame
#         face2 = gray[y:y+h, x:x+w]
        
#         # Resize the face to the standard size
#         face2 = cv2.resize(face2, (64, 64))
#         cv2.imshow("face", face1)
#         cv2.imshow("face2", face2)

#         # Compute the mean squared error between the two faces
#         mse = np.mean((face1 - face2) ** 2)
#         print(mse)

#         # If the MSE is below a certain threshold, conclude that the faces are the same
#         if mse < 110:
#             text = "Same Person"
#             color = (0, 255, 0)
#         else:
#             text = "Different Person"
#             color = (0, 0, 255)

#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

#         # Put the text on the frame
#         cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     # Display the frame
#     cv2.imshow("Face Comparison", frame)

#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the camera
# camera.release()

# # Close all windows
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load the reference image
reference_image_benjamin = cv2.imread("images/b.jpg", 0)
cv2.resize(reference_image_benjamin, (64, 64))
reference_image_ronja = cv2.imread("images/r.jpg", 0)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier("haarcascade.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # For each face detected
    for (x, y, w, h) in faces:

        # Extract the face area and resize it
        face_area = gray[y:y+h, x:x+w]
        face_area = cv2.resize(face_area, (reference_image_benjamin.shape[1], reference_image_benjamin.shape[0]))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        # Compare the face area with the reference image
        result_b = cv2.matchTemplate(face_area, reference_image_benjamin, cv2.TM_CCOEFF_NORMED)
        result_r = cv2.matchTemplate(face_area, reference_image_ronja, cv2.TM_CCOEFF_NORMED)
        # Check if the comparison result is above a certain threshold
        threshold = 0.4
        loc_b = np.where(result_b >= threshold)
        loc_r = np.where(result_r >= threshold)
        
        
        
        print(loc_b[0])
        # If it is a match, draw a rectangle around the face and display the result
        if len(loc_b[0]) > 0:
            print("Benjamin:" + str(loc_b[0]))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Benjamin", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif len(loc_r[0]) > 0:
            print("Ronja:" + str(loc_r[0]))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(frame, "Ronja", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
    # Display the result
    cv2.imshow("Result", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
