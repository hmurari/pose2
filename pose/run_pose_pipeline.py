import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

import time
import argparse
from pose import PosePre, PoseEngine, PosePost, PoseDraw
import sys

UTILS_PATH = '/utils'
ENGINE_PATH = '/pose/generated/densenet121_baseline_att_trt.pth'

sys.path.append(UTILS_PATH)

from video import Video
from display import Display
from pipeline import Pipeline, EOS


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--loop', action="store_true")
    parser.add_argument('--codec', type=str, default='h264')
    args = parser.parse_args()

    # media
    video = Video(args.video, args.width, args.height, loop=args.loop, codec=args.codec)
    #display = Display(args.width, args.height)

    # pose
    pre = PosePre()
    engine = PoseEngine(ENGINE_PATH)
    post = PosePost()
    draw = PoseDraw()

    def video_f(data):
        data['image'] = video.read()

        if data['image'] is None:
            data = EOS
            video.destroy()

        return data

    def pre_f(data):
        if data is EOS:
            return data
        data['tensor'] = pre(data['image'])
        return data

    def engine_f(data):
        if data is EOS:
            return data
        data['cmap'], data['paf'] = engine(data['tensor'])
        return data

    def post_f(data):
        if data is EOS:
            return data
        data['counts'], data['objects'], data['peaks'] = post(data['cmap'], data['paf'])
        return data

    def draw_f(data):
        if data is EOS:
            return data
        draw(data['image'], data['counts'], data['objects'], data['peaks'])
        return data

    then = None

    def display_f(data):
        global then

        if data is EOS:
            video.destroy_window()
            return data

        # display
        video.show_frame(data['image'])

        # print time delta
        now = time.time()
        if then is not None:
            print('FPS: %f' % (1.0 / (now - then)))
        then = now

        return data

    pipeline = Pipeline([
        video_f,
        pre_f,
        engine_f,
        post_f,
        draw_f,
        display_f
    ])

    pipeline.start()
