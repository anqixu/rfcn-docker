#!/usr/bin/env python

"""
Derived from py-faster-rcnn/tools/demo.py
"""

import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
TARGET_CLASSES = ('bicycle', 'bus', 'car', 'motorbike')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def process(net, im_file, net_name):
    """Detect object classes in an image using pre-computed object proposals."""
    
    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Draw image on virtual plot canvas
    im = im[:, :, (2, 1, 0)] # swap RGB to BGR
    #fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(im, aspect='equal')

    # Visualize detections for TARGET_CLASSES
    CONF_THRESH = 0.2
    NMS_THRESH = 0.4
    recog_counts = dict()
    for cls_ind, cls in enumerate(CLASSES):
        if cls not in TARGET_CLASSES:
          continue
        
        # Filter detections
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        # Draw detection bounding boxes
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        recog_counts[cls] = len(inds)
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '%s %d%%' % (cls, score*100),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    title_str = 'model=%s, min_conf=%.1f, nms_thresh=%.1f\ncounts: %s' % (net_name, CONF_THRESH, NMS_THRESH, str(recog_counts))
    ax.set_title(title_str)

    # Save plot
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    im_dir, im_base = os.path.split(im_file)
    im_basename, im_ext = os.path.splitext(im_base)
    dest_path = os.path.join(im_dir, '%s.detections%s' % (im_basename, im_ext))
    plt.savefig(dest_path, bbox_inches='tight')
    print title_str
    print 'Saved to ' + dest_path

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN ParQ Processing Script')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('images_path', help='Path to image file to be processed (e.g. ./data/demo/004545.jpg)')

    args = parser.parse_args()

    return args

def resolve_file_path(path):
  if len(path) > 0:
    if path[0] == '.' or path[0] == '/' or path[0] == '~':
      path = os.path.abspath(path)
    else:
      path = os.path.join(os.getcwd(), path)
    if os.path.isfile(path):
      return path
  return None


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
    # Assume single image file; resolve path
    if len(args.images_path) > 0 and os.path.isdir(args.images_path):
      print 'TODO: implement per-dir processing'
      sys.exit(-1)
    image_path = resolve_file_path(args.images_path)
    if image_path is None:
      print 'Could not resolve or locate images_path: %s' % args.images_path
      sys.exit(-1)

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))


    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    process(net, image_path, args.demo_net)
