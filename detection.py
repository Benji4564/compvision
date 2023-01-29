
import cv2

# Start videocapture
cap = cv2.VideoCapture(0)

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found = False
while(not found):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect humans in the frame
    humans, _ = hog.detectMultiScale(frame)

    # Draw rectangles around detected humans
    for (x, y, w, h) in humans:
        found = True

    # Display the resulting frame
    # cv2.imshow('Video Capture',frame)

    # Wait for 'q' key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("done")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

