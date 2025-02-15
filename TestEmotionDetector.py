from tensorflow.keras.models import model_from_json
import numpy as np
import cv2

# Emotion dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ✅ Load Model Architecture
json_path = "C:\\Users\\Asus\\Downloads\\Emotion_detection_with_CNN\\Emotion_detection_with_CNN-main\\model\\emotion_model.json"
weights_path = "C:\\Users\\Asus\\Downloads\\Emotion_detection_with_CNN\\Emotion_detection_with_CNN-main\\model\\emotion_model.weights.h5"

try:
    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # ✅ Load Weights
    model.load_weights(weights_path)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize OpenCV for video processing
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = roi_gray / 255.0  # Normalize pixel values

        prediction = model.predict(roi_gray)
        max_index = np.argmax(prediction[0])
        emotion = emotion_dict[max_index]

        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
