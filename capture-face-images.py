import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('data', 'images')
number_images = 30

def capture_images():
    cap = cv2.VideoCapture(0)
    for imgnum in range(number_images):
        print(f"Collecting image {imgnum}")
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, f"{str(uuid.uuid1())}.jpg")
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.5)
        print(f"Created file {imgname}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    capture_images()

if __name__ == "__main__":
    main()