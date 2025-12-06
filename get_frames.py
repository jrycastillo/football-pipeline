import cv2
cap = cv2.VideoCapture("121364_0.mp4")
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()
