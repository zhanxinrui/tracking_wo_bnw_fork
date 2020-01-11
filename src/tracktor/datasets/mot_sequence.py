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


class MOT17_Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

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

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det')
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT17Labels')
        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        assert seq_name in self._train_folders or seq_name in self._test_folders, \
            'Image set does not exist: {}'.format(seq_name)

        self.data, self.no_gt = self._sequence()

    def __len__(self):
        return len(self.data)

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

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
            label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
            mot17_label_path = osp.join(self._mot17_label_dir, 'train')
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)
            label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
            mot17_label_path = osp.join(self._mot17_label_dir, 'test')
        raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []
        train = []
        val = []

        visibility = {}
        boxes = {}
        dets = {}

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []

        no_gt = False
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

        for i in range(1,seqLength+1):
            im_path = osp.join(imDir,"{:06d}.jpg".format(i))

            sample = {'gt':boxes[i],
                      'im_path':im_path,
                      'vis':visibility[i],
                      'dets':dets[i],}

            total.append(sample)

        return total, no_gt

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        if self._dets == "DPM":
            det_file = osp.join(label_path, 'det', 'det.txt')
        elif self._dets == "DPM_RAW16":
            det_file = osp.join(raw_label_path, 'det', 'det-dpm-raw.txt')
        elif "17" in self._seq_name:
            det_file = osp.join(
                mot17_label_path,
                f"{self._seq_name}-{self._dets[:-2]}",
                'det',
                'det.txt')
        else:
            det_file = ""
        return det_file

    def __str__(self):
        return f"{self._seq_name}-{self._dets[:-2]}"

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if "17" in self._dets:
            file = osp.join(output_dir, 'MOT17-'+self._seq_name[6:8]+"-"+self._dets[:-2]+'.txt')
        else:
            file = osp.join(output_dir, 'MOT16-'+self._seq_name[6:8]+'.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])


import pycocotools.mask as rletools
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
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT17Labels')
        self._mask_label_dir = osp.join(cfg.DATA_DIR,'MOTS17+') #mask_label_dir
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

    def write_results_files(self, results, output_dir):
        """Write the detections in the format for MOT17Det sumbission

        all_boxes[image] = N x 5 array of detections in (x1, y1, x2, y2, score)

        Each file contains these lines:
        使用detetion的bbox保存文件:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>,<x>,<y>,<z>
        使用ｍask 的bbox保存文件:
        为了和mots数据集统一：
        MOTS标签为：
        <frame>,<id>,<class_id> <img_height> <img_width>  <rle>
        <frame>,-1,-1,<img_height> <img_width>  <rle> <x1>, <y1>,<x2>,<y2>,<x3>,<y3>,<x4>,<y4>


        Files to sumbit:
        ./MOT17-01.txt
        ./MOT17-02.txt
        ./MOT17-03.txt
        ./MOT17-04.txt
        ./MOT17-05.txt
        ./MOT17-06.txt
        ./MOT17-07.txt
        ./MOT17-08.txt
        ./MOT17-09.txt
        ./MOT17-10.txt
        ./MOT17-11.txt
        ./MOT17-12.txt
        ./MOT17-13.txt
        ./MOT17-14.txt
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        files = {}
        mask_files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img1, name = osp.split(path)
            # get image number out of name
            frame = int(name.split('.')[0])
            # smth like /train/MOT17-09-FRCNN or /train/MOT17-09
            tmp = osp.dirname(img1)
            # get the folder name of the sequence and split it
            tmp = osp.basename(tmp).split('-')
            # Now get the output name of the file
            out = tmp[0] + '-' + tmp[1] + '.txt'  # 将一个seq的结果放到一个文件中，所以只需要'MOT17' + '-' + '01' +'.txt'
            outfile = osp.join(output_dir, out)
            mask_outfile = osp.join(output_dir, 'mask', out)

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []
            if mask_outfile not in files.keys():
                files[mask_outfile] = []
            # 放到一个字典中，索引是地址，val是一个列表表示里面应该有的行
            for box, score in zip(res['boxes'], res['scores']):
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append([frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
            for mask in res['masks']:
                box = mask_reader.mask_to_bbox(mask)
                if box == None:
                    continue
                rle_mask = rletools.encode(mask)
                mask_width = mask.shape[0]
                mask_height = mask.shape[1]
                line = []
                line.append(frame)
                line.append(-1)
                line.append(-1)
                line.append(mask_height)
                line.append(mask_width)
                line.append(rle_mask+1)  # 需要根据height和width和rle恢复mask
                # 因为很可能不平行于坐标轴,所以保存四个点
                for v in box:
                    line.append(v[0])
                    line.appned(v[1])

                files[mask_outfile].append(line)

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def _sequence(self):
        seq_name = self._seq_name#MOT17-02
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
            label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
            mask_label_path = osp.join(self._mask_label_dir, 'train')
            mot17_label_path = osp.join(self._mot17_label_dir, 'train')

        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)
            label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
            mask_label_path = osp.join(self._mask_label_dir, 'test')
            mot17_label_path = osp.join(self._mot17_label_dir, 'test')
        raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])

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

        gt_mask_seq = mask_reader.load_txt(mask_gt_file)
        gt_masks = rletools.decode(gt_mask_seq)
        gt_masks = torch.from_numpy(gt_masks)

        if osp.exists(mask_label_path):
            with open(mask_label_path,"r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    frame = row[0]
                    img_height = row[3]
                    img_width = row[4]
                    rle = row[5]
                    mask_box = np.array([[row[6]-1,row[7]-1],[row[8]-1,row[9]-1],[row[10]-1,row[11]-1],[row[12]-1,row[13]-1]])
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
class MOT19CVPR_Sequence(MOT17_Sequence):

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

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT19_CVPR')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT19_CVPR')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = Compose([ToTensor(), Normalize(normalize_mean,
                                                         normalize_std)])

        if seq_name:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data = self._sequence()
        else:
            self.data = []

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        # FRCNN detections
        if "CVPR19" in self._seq_name:
            det_file = osp.join(mot17_label_path, self._seq_name, 'det', 'det.txt')
        else:
            det_file = ""
        return det_file

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


class MOT17LOWFPS_Sequence(MOT17_Sequence):

    def __init__(self, split, seq_name=None, dets='', vis_threshold=0.0,
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

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17_LOW_FPS', f'MOT17_{split}_FPS')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT17_LOW_FPS', f'MOT17_{split}_FPS')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = Compose([ToTensor(), Normalize(normalize_mean,
                                                         normalize_std)])

        if seq_name:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data = self._sequence()
        else:
            self.data = []
