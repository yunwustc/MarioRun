"""
This is the main file to run the Mario.
1. load the trained model
2. run the video recording
3. run the Mario

@Author: Yun Wu
@Email: yunwustc@gmail.com
"""

import cv2
import multiprocessing as _mp
from src.utils import mario
from src.model import load_model, predict


def main():
    model = load_model(model_path=r'./src/Mario_model')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()
    process = mp.Process(target=mario, args=(v, lock))
    process.start()
  
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 100)
    fontScale = 2
    color = (255, 0, 0)
    thickness = 2

    while True:
        key = cv2.waitKey(10)
        if key == 27:
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        category = predict(model, frame)
	
        if category:
            if category == 'Run_Right':
                action = 1
            elif category == 'Jump_Right':
                action = 2
            elif category == 'Jump':
                action = 5
            elif category == 'Run_Left':
                action = 6
            elif category == 'Jump_Left':
                action = 7
            elif category == 'Stay':
                action = 0
            else:
                action = 0
	
            with lock:
                v.value = action

        cv2.putText(frame, category, org, font, fontScale, color, thickness)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()
    process.terminate()

if __name__ == '__main__':
    main()
