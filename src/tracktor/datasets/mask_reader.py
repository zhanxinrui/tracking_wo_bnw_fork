import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import os
import cv2
import matplotlib.pyplot as plt


# 用来获取分割的数据
class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


# 得到所有seq中每一帧中的分割和检测
def load_all_sequences(path, seqmap):
    objects_per_frame_per_sequence = {}
    for seq in seqmap:
        # print("Loading sequence", seq)
        # print('path',path)
        # print('seqmap',seqmap)
        seq_path_folder = os.path.join(path, seq)
        seq_path_txt = os.path.join(path, seq + ".txt")
        # print("seq_path_txt",seq_path_txt)
        if os.path.isdir(seq_path_folder):
            objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
        elif os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
        else:
            assert False, "Can't find data in directory " + path

    return objects_per_frame_per_sequence



def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]
            # print('width',int(fields[3]))
            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            # img = rle2mask(fields[5],int(fields[4]),int(fields[3]))
            # img = fields[5].encode(encoding='UTF-8').decode(encoding='UTF-8')
            # bit_mask = rletools.decode(mask)
            # print(img)
            # import cv2 as cv
            # # src = cv.imread()
            # cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
            # cv.imshow('input_image', img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # 将同一张图片的分割放到一张图片上
            # if frame not in combined_mask_per_frame:
            #   combined_mask_per_frame[frame] = mask
            # elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
            #   assert False, "Objects with overlapping masks in frame " + fields[0]
            # else:
            #   combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)

            # objects_per_frame[frame].append(SegmentedObject(
            #   mask,
            #   class_id,
            #   int(fields[1])
            # ))
            objects_per_frame[frame].append(mask)

        # return objects_per_frame
        # return combined_mask_per_frame
        return objects_per_frame


def load_inference_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]
            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            objects_per_frame[frame].append(mask)

        return objects_per_frame

def load_images_for_folder(path):
    files = sorted(glob.glob(os.path.join(path, "*.png")))

    objects_per_frame = {}
    for file in files:
        objects = load_image(file)
        frame = filename_to_frame_nr(os.path.basename(file))
        objects_per_frame[frame] = objects

    return objects_per_frame


def filename_to_frame_nr(filename):
    assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
    return int(filename.split('.')[0])


def load_image(filename, id_divisor=1000):
    img = np.array(Image.open(filename))
    obj_ids = np.unique(img)

    objects = []
    mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
    for idx, obj_id in enumerate(obj_ids):
        if obj_id == 0:  # background
            continue
        mask.fill(0)
        pixels_of_elem = np.where(img == obj_id)
        mask[pixels_of_elem] = 1
        objects.append(SegmentedObject(
            rletools.encode(mask),
            obj_id // id_divisor,
            obj_id
        ))
        # print(mask)
    return objects


# 得到所有的seg的帧编号列表
def load_seqmap(seqmap_filename):
    print("Loading seqmap...")
    seqmap = []
    max_frames = {}
    with open(seqmap_filename, "r") as fh:
        for i, l in enumerate(fh):
            fields = l.split(" ")
            seq = "%04d" % int(fields[0])
            seqmap.append(seq)
            max_frames[seq] = int(fields[3])
    return seqmap, max_frames


def write_sequences(gt, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for seq, seq_frames in gt.items():
        write_sequence(seq_frames, os.path.join(output_folder, seq + ".txt"))
    return


def write_sequence(frames, path):
    with open(path, "w") as f:
        for t, objects in frames.items():
            for obj in objects:
                print(t, obj.track_id, obj.class_id, obj.mask["size"][0], obj.mask["size"][1],
                      obj.mask["counts"].decode(encoding='UTF-8'), file=f)


def mask_to_bbox(mask,mask_thresh):
    target_mask = (mask > mask_thresh).astype(np.uint8)
    contours, hierachy = cv2.findCOntours(target_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_TC89_L1)  # EXTERNAL使用最外层的contour,CHAIN_APPROX_NONE储存所有的点
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    contour = contours[np.argmax(cnt_area)]
    polygon = contour.reshape(-1, 2)  # 二维的点，统一成两列的格式
    prbox = cv2.boxPoints(cv2.minAreaRect(polygon))


# if __name__ == '__main__':
#     seg_root = '/home/hook/Downloads/Dataset/MOT/trackrcnn/MOTS17/instances_txt'
#     # seg_root = '/home/hook/Downloads/Dataset/MOT/trackrcnn/MOTS17/instances'
#     seg_seq_dir = ["%04d" % idx for idx in [2, 5, 9, 11]]
#     # for idx in seg_seq_dir:
#     #     seq_dir = os.path.join(seg_root,idx+'.txt')
#     #     seq_map,max_frames = load_seqmap('/home/hook/Downloads/Dataset/MOT/trackrcnn/MOTS17/instances_txt/0002.txt')
#     #     obj = load_sequences(seq_dir,seq_map)
#     # for idx in seg_seq_dir:
#     #     seq_dir = os.path.join(seg_root,idx+'.txt')
#     #     seq_map,max_frames = load_seqmap('/home/hook/Downloads/Dataset/MOT/trackrcnn/MOTS17/instances_txt/0002.txt')
#     #     obj = load_sequences(seq_dir,seq_map)
#     # print(seq_map)
#     mask_seq = load_sequences(seg_root, seg_seq_dir)
#     bit_mask = rletools.decode(mask_seq['0002'][1])
#     print(bit_mask)