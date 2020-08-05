import time
import argparse
from pose import PosePre, PoseEngine, PosePost, PoseDraw
import sys
import pprint

UTILS_PATH = '/utils'
ENGINE_PATH = '/pose/generated/densenet121_baseline_att_trt.pth'

sys.path.append(UTILS_PATH)

from video import Video
from display import Display

# Whether to record the output video or not.
record = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--codec', type=str, default='h264')
    parser.add_argument('--loop', type=str, default='')
    args = parser.parse_args()

    # media
    video = Video(None, args.width, args.height, loop=args.loop, codec=args.codec)
    #display = Display(args.width, args.height)

    # pose
    pre = PosePre()
    engine = PoseEngine(ENGINE_PATH)
    post = PosePost()
    draw = PoseDraw()

    t0 = time.time()

    while True:

        # read image from file
        image = video.read()

        # preprocess image (resize, send to GPU, normalize, permute dims)
        tensor = pre(image)

        # run pose engine
        cmap, paf = engine(tensor)

        # parse objects
        counts, objects, peaks = post(cmap, paf)

        # draw objects on image
        draw(image, counts, objects, peaks)


        # render image to display
        ret = video.show_frame(image, record=record)
        if ret == -1:
            break

        # print FPS
        t1 = time.time()
        # print('FPS: %f' % (1.0 / (t1 - t0)))
        t0 = t1
        
        #print('- - - - - - - - - - - - - - - - - - -')
        #print('Objects found: {}'.format(counts))
        #print('Objects: ')
        #pprint.pprint(objects)
        #print('- - - - - - - - - - - - - - - - - - -')
    
    video.destroy()
    
