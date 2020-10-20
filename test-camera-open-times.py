import cv2
import time

def main():
    start_time = time.time()
    print('Opening camera')
    
    cap = cv2.VideoCapture('/dev/video0')
    open_time = time.time()
    print('Open Time taken: {}'.format(open_time - start_time))

    ret, frame = cap.read()
    read_time = time.time()
    print('1st Read Time taken: {}'.format(read_time - open_time))

    ret, frame = cap.read()
    read2_time = time.time()
    print('2nd Read Time taken: {}'.format(read2_time - read_time))
    
if __name__ == '__main__':
    main()

