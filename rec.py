import numpy as np
import cv2
import mtcnn
import keras.models as models

# Load the MTCNN face detector
detector = mtcnn.MTCNN()

# Load the FaceNet model
model = models.load_model('facenet_keras.h5')

# Define a function to extract face embeddings from an image
def get_embedding(model, face_pixels):
    # Scale pixel values
    face_pixels = face_pixels.astype('float32')
    # Standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # Make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Load the first face to compare
pixels1 = cv2.imread("face1.jpg")

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Loop over the faces
    for face in faces:
        x, y, w, h = face['box']
        face_pixels = frame[y:y+h, x:x+w]
        
        # Extract face embeddings for the current face
        embedding = get_embedding(model, face_pixels)
        
        # Compare the embeddings
        dist = np.sum(np.square(embedding - get_embedding(model, pixels1)))
        if dist < 0.6:
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

    #
