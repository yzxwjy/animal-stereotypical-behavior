from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', default='D:/yzx/siamban-master/experiments/siamban_r50_l234/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='D:/yzx/siamban-master/experiments/siamban_r50_l234/model.pth', type=str, help='model name')
parser.add_argument('--video_name', default='D:/yzx/siamban-master/demo/2.mp4', type=str,
                    help='videos or image files')
parser.add_argument('--save', action='store_true', default='True',
                    help='whether visualzie result')
######help='whether visualzie result'
args = parser.parse_args()

# def __init__(self, track_id, n_init, max_age, feature=None):
# self.track_id = track_id

dict_box = dict()
frame_num = 0


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4') or \
            video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        global frame_num
        while True:
            ret, frame = cap.read()
            if ret:
                frame_num = frame_num + 1
                yield frame
                # print(frame_num)
            else:
                break

    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    # model.eval().to(device)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # model_name = args.snapshot.split('/')[-1].split('.')[0]

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)   ###窗体名称

    for frame in get_frames(args.video_name):
        if frame_num == 1:
            # build video writer
            if args.save:
                if args.video_name.endswith('avi') or \
                        args.video_name.endswith('mp4') or \
                        args.video_name.endswith('mov'):
                    cap = cv2.VideoCapture(args.video_name)
                    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30

                save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0])  # (w, h)
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)  ###tupianzhuanshipin
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            # frame_num=0
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
                a = (bbox[0] + bbox[0] + bbox[2]) / 2
                b = (bbox[1] + bbox[1] + bbox[3]) / 2
                center = (int(a), int(b))

                # pts = deque(maxlen=315)
                # pts.append(center)
                # for i in range(1, len(pts)):
                # if pts[i - 1] is None or pts[i] is None:
                # continue
                # thickness = int(np.sqrt(315 / float(i + 1) * 2.5))
                # cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness=2, linetype=8)
                # cv2.imshow(video_name, frame)

                # center = [a, b]

                id = frame_num
                dict_box.setdefault(id, []).append(center)
                # print(dict_box)
                # cv2.circle(frame, (int(a), int(b)), 7, (255, 255, 255), -1)
                key_list = list(dict_box.keys())
                value_list = list(dict_box.values())
                # print

                if frame_num > 2:
                    # index_start = len(value_list) - 5
                    # index_end = index_start + 4
                    # value1 = value_list[index_start]
                    # value2 = value_list[index_end]
                    # cv2.line(frame, value1[0], value2[0], (0, 255, 0), thickness=10, lineType=8)
                    for c in range(len(value_list) - 2):
                        index_start = c
                        index_end = index_start + 1
                        value1 = value_list[index_start]
                        value2 = value_list[index_end]
                        # pt1 = value1[0]
                        # pt2 = value2[0]
                        cv2.line(frame, value1[0], value2[0], (0, 255, 0), thickness=5, lineType=8)
                    # print(value_list[frame_num - 1])
                    # for keys, values in dict_box.items():
                    # for c in range(len(values) - 1):
                    # index_start = c
                    # index_end = index_start + 1
                    # cv2.line(frame_num, tuple(map(int, values[index_start])), tuple(map(int, values[index_end])),
                    # (0, 255, 0), thickness=2, linetype=8)


                file_path = 'D:/yzx/siamban-master/2.txt'
                with open(file_path, 'a') as f:
                    f.writelines('\n' + str(a) + ',' + str(b))

                # print(' '+str(a)+','+str(b))

            cv2.imshow(video_name, frame)
            cv2.waitKey(40)

        if args.save:
            video_writer.write(frame)

    if args.save:
        video_writer.release()

if __name__ == '__main__':
    main()
