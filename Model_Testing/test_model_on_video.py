import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# -------------------------------
# 1. Load the Trained Model
# -------------------------------
model_path = "fine_tuned_mobilenetv2.h5"
model = tf.keras.models.load_model(model_path)

# Define the class names (ensure these match the order used during training)
class_names = [
    "adho_mukh_svanasana",
    "ashtanga_namaskara",
    "ashwa_sanchalanasana",
    "bhujangasana",
    "hasta_utthanasana",
    "kumbhakasana",
    "padahastasana",
    "pranamasana"
]

# -------------------------------
# 2. Setup MediaPipe Pose
# -------------------------------
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# 3. Setup OpenCV for Video File Capture
# -------------------------------
video_path = "Downloads\\VID-20250309-WA0014.mp4"  # Update with your video file path
cap = cv2.VideoCapture(video_path)

# -------------------------------
# 4. Frame Grouping Settings
# -------------------------------
group_size = 5  # Number of frames to group together
frame_buffer = []  # Buffer to store frames for grouping

# Variables to hold last computed prediction
last_label = None
last_confidence = None

# -------------------------------
# 5. Video Pose Recognition Loop (Grouped Frames)
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with MediaPipe Pose for landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Preprocess frame for model input:
    # Resize frame to match the model's expected input (224x224)
    frame_resized = cv2.resize(frame, (224, 224))
    # Convert BGR -> RGB and normalize to [0, 1]
    input_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
    frame_buffer.append(input_frame)

    # When we have enough frames in the buffer, predict on the batch
    if len(frame_buffer) == group_size:
        # Convert list to numpy array (shape: (group_size, 224, 224, 3))
        batch = np.array(frame_buffer)
        predictions = model.predict(batch)
        # Average predictions over the group (resulting in a vector of shape (num_classes,))
        avg_prediction = np.mean(predictions, axis=0)
        label_index = int(np.argmax(avg_prediction))
        last_confidence = avg_prediction[label_index]
        last_label = class_names[label_index]

        # Slide the buffer: remove the first half of the frames for overlapping windows
        frame_buffer = frame_buffer[group_size // 2:]

    # If we have a computed label, display it
    if last_label is not None:
        cv2.putText(frame, f"{last_label}: {last_confidence:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show the video frame with the pose prediction and landmarks
    cv2.imshow("Video Pose Recognition (Grouped Frames)", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 6. Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
