import numpy as np
import cv2
from insightface.app import FaceAnalysis


if __name__ == '__main__':
  ESC_KEY = 27
  INTERVAL= 33
  FRAME_RATE = 30
  WINDOW_NAME = "gaussian"

  DEVICE_ID = 0

  cap = cv2.VideoCapture(DEVICE_ID)

  end_flag, c_frame = cap.read()
  height, width, channels = c_frame.shape

  cv2.namedWindow(WINDOW_NAME)

  app = FaceAnalysis()
  app.prepare(ctx_id=0, det_size=(640, 640))

  # 変換処理ループ
  while end_flag == True:
    img = c_frame
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = app.get(np.asarray(img))
    rimg = app.draw_on(img, faces)
    cv2.imshow(WINDOW_NAME, rimg)
    key = cv2.waitKey(INTERVAL)

    if key == ESC_KEY:
      break

    end_flag, c_frame = cap.read()

  cv2.destroyAllWindows()
  cap.release()
  