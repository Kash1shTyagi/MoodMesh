from services.emotion.detection import FaceDetector
import cv2

config = {
  "model_path": "data/models/buffalo_l/det_10g.onnx",
  "min_confidence": 0.5,
  "max_faces": 5,
  "input_size": [320, 240],
  "landmark_points": 5
}
detector = FaceDetector(config)

img = cv2.imread("test_face.jpg")
faces = detector.detect(img)
print(f"Found {len(faces)} faces")
for f in faces:
  x,y,w,h = f["box"]
  cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
  for (lx,ly) in f["landmarks"]:
    cv2.circle(img, (lx,ly), 3, (0,0,255), -1)
cv2.imshow("det", img); cv2.waitKey()
