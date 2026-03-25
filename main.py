import sys
print("Dang chay Python phien ban:", sys.version)
import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import math   
from openyt import open_youtube


# Hàm hỗ trợ phát hiện chớp mắt
def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def calculate_ear(eye_point, landmarks, img_w, img_h):
    coords = [(int(landmarks.landmark[p].x * img_w), int(landmarks.landmark[p].y * img_h)) for p in eye_point]
    v1 = euclidean_distance(coords[1], coords[5])
    v2 = euclidean_distance(coords[2], coords[4])
    h = euclidean_distance(coords[0], coords[3])
    return (v1 + v2) / (2.0 * h)


# Khởi tạo các mô hình 
recog_tool = cv.face.LBPHFaceRecognizer_create()
recog_tool.read("face_recognizer_model.yml") #đọc mô hình đã huấn luyện
label_dict = np.load("label_dict.npy", allow_pickle=True).item()
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml") 

## Mô hình MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
#Thay đổi max_num_faces để nhận diện nhiều khuôn mặt cùng lúc
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.20    # Ngưỡng chớp mắt, có thể điều chỉnh tùy theo thực tế



cap = cv.VideoCapture(0)
check = False
is_real_person = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Loi camera!")
        break
    if frame is not None:
      h, w, _ = frame.shape
      #Kiểm tra chớp mắt cho tất cả khuôn mặt được phát hiện
      rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
      results = face_mesh.process(rgb_frame)

      if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
              left_ear = calculate_ear(LEFT_EYE, face_landmarks, w, h)
              right_ear = calculate_ear(RIGHT_EYE, face_landmarks, w, h)
              avg_ear = (left_ear + right_ear) / 2.0

              if avg_ear < EAR_THRESHOLD:
                  is_real_person = True
      status_text = "Phat hien nguoi that" if is_real_person else "Khong phai nguoi that/La anh tinh"
      status_color = (0, 255, 0) if is_real_person else (0, 0, 255)
      cv.putText(frame, status_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

      #Nhận diện bằng mô hình LBPH
      if is_real_person:
          gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
          dgray = cv.GaussianBlur(gray, (5, 5), 0)
          faces = face_cascade.detectMultiScale(dgray, 1.3, 5)

          for (x, y, w, h) in faces:
              face_img = dgray[y:y+h, x:x+w]
              name_id, dotincay = recog_tool.predict(face_img)

              if dotincay < 50:
                  name = label_dict[name_id]
                  color = (0, 255, 0)

                  if not check:
                      check = True
                      open_youtube()
              else: 
                  name = "Unknown"
                  color = (0, 0, 255)

              cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
              cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
