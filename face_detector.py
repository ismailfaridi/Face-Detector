import cv2 as cv

# Load Haar-Cascasde Models
face_model = cv.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")
eye_model = cv.CascadeClassifier("./haarcascade/haarcascade_eye.xml")
smile_model = cv.CascadeClassifier("./haarcascade/haarcascade_smile.xml")

if face_model.empty() or eye_model.empty() or smile_model.empty():
    print("ERROR: Model is NOT loaded.")
    exit()

# Initialize Internal Webcam
cap = cv.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Unable to read frame from webcam.")
        break

    # Flip the frame to correctly display the image, and convert it into grayscale for model feeding
    frame = cv.flip(frame, 1)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_model.detectMultiScale(gray_img, 1.1, 25)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ROI (region of interest) - Crop the detected face, and find eyes & smile in it
        roi_gray_img = gray_img[y:y+h, x:x+w]

        eyes = eye_model.detectMultiScale(roi_gray_img, 1.3, 10)
        smile = smile_model.detectMultiScale(roi_gray_img, 1.3, 20)

        if len(eyes) > 0:
            cv.putText(frame, "Eyes Detected", (x, y-5), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        if len(smile) > 0:
            cv.putText(frame, "Smile Detected", (x, y-20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # Show Detection Results
    cv.imshow("Live Detection", frame)
    # Wait 1ms for user to press 'q' to exit the app
    if (cv.waitKey(1) & 0xFF) == ord('q'):
        print("Successfully Exit.")
        break

cap.release()

cv.destroyAllWindows()
