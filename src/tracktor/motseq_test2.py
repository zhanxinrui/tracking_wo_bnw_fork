import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2

from ..config import cfg
from torchvision.transforms import ToTensor
import mask_reader

class MOTS17_Sequence(MOT17_Sequence):


    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):

        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """


        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold
        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det')#放图片
        self._mots17_dir = osp.join(cfg.DATA_DIR,'MOTS17')#mots官方数据集
        self._mots17plus_dir = osp.join(cfg.DATA_DIR,'MOTS17+')#修改后的数据集
        self._seg_train_img_folders = os.listdir(os.path.join(self._mots17_dir),'instances')
        self._seg_train_txt_folders = os.listdir(os.path.join(self._mots17_dir),'instances_txt')

        self._label_dir = osp.join(cfg.DATA_DIR,'MOTS17+')
        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))#图片位置
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        assert seq_name in self._train_folders or seq_name in self._test_folders, \
            'Image set does not exist: {}'.format(seq_name)

        self.data, self.no_gt = self._sequence()

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample


    def write_results(self, all_tracks, output_dir):
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, f'{self._seq_name}.txt')

        print("[*] Writing to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])


    def _sequence(self):
        seq_name = self._seq_name#MOT17-02
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
            # label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
            label_path = osp.join(self._label_dir, 'train')
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)
            # label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
            label_path = osp.join(self._label_dir, 'test')
        # raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')#det 的gt
        # gt_file = osp.join(label_path,)
        mask_gt_file = osp.join(self._seg_train_txt_folders,"%04d"%int(seq_name[-2:])+'.txt' )

        total = []
        train = []
        val = []

        visibility = {}
        boxes = {}
        dets = {}
        masks = {}
        mask_boxes = {}

        for i in range(1, seqLength+1):
            boxes[i] = {}   #暂时不懂意图
            visibility[i] = {}
            dets[i] = []
            masks[i] = []
            mask_boxes[i] = [] 

        no_gt = False
        #这里以一个seq进行加载
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(row[4]) - 1
                        y2 = y1 + int(row[5]) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        det_file = self.get_det_file(label_path, raw_label_path, mot17_label_path)

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                    dets[int(row[0])].append(bb)

        gt_mask_seq = load_txt(mask_gt_file)
        gt_masks = rletools.decode(mask_seq)
        gt_masks = torch.from_numpy(mask)

        if osp.exists(label_path):
            with open(label_path,"r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    frame = row[0]
                    img_height = row[3]
                    img_width = row[4]
                    rle = row[5]
                    mask_box = np.array([[row[6],row[7]],[row[8],row[9]],[row[10],row[11]],[row[12],row[13]]])
                    mask = {'size': [int(img_height), int(img_width)], 'counts': rle.encode(encoding='UTF-8')}
                    mask = rletools.decode(mask) #恢复成矩阵
                    masks[frame].append(mask)
                    mask_boxes[frame].append(mask_box)

        for i in range(1,seqLength+1):
            im_path = osp.join(imDir,"{:06d}.jpg".format(i))

            sample = {'gt_box':boxes[i],
                      'im_path':im_path,
                      'vis':visibility[i],
                      'dets':dets[i],
                      'gt_mask':gt_masks[i],
                      'mask': masks[i],
                      'mask_box':mask_boxes[i]
                      }


            total.append(sample)



        return total, no_gt