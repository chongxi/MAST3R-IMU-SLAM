import cv2

# Open the camera
cap = cv2.VideoCapture('/dev/video4')

# Set the stream properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Capture frame continuously
    ret, frame = cap.read()

    if ret:
        # Display the frame
        cv2.imshow('Camera Feed', frame)
    else:
        print("Failed to capture frame")
        break

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
