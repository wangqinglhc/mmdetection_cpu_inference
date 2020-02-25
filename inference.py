import argparse
import os, glob
import cv2

from mmdet.apis import init_detector, show_result

# -- label names to be shown on the inferenced pictures
label_class = {1: 'HP', 2: 'ST', 3: 'UP'}

# -- confidence score threshold 
score_thr = 0.5

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a folder of pics')
    parser.add_argument('config', help='config file for training the detector')
    parser.add_argument('checkpoint', help='the trained model file')
    parser.add_argument('img_folder', help='input folder for the images')
    parser.add_argument('output_folder', help='output folder for the inferenced images')
    parser.add_argument('--output_bbox_file', help='bbox info for further calculation')
    parser.add_argument('--post_processing', action='store_true', help='remove duplicate bboxes')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if not os.path.exists(args.config):
        config_file = './configs/cascade_rcnn_x101_64x4d_fpn_1x.py'
    else:
        config_file = args.config

    assert os.path.exists(args.checkpoint), 'you need a trained model file!'
    checkpoint = args.checkpoint

    assert os.path.exists(args.img_folder), 'input folder does not exist'
    img_folder = args.img_folder

    if not os.path.exists(args.output_folder):
        try:
            os.makedirs(args.output_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    output_folder = args.output_folder

    model = init_detector(config_file, checkpoint)
    img_files = glob.glob(os.path.join(img_folder, '*.jpg'))
    img_files += glob.glob(os.path.join(img_folder, '*.png'))
    if args.output_bbox_file:
        file_bbox = open(args.output_bbox_file, 'w')
    else:
        file_bbox = None

    for img in img_files:
        print('Run inference on: ', img)
        bboxes, labels = model.detect(img)
        output_img_filename = os.path.join(output_folder, img.split('/')[-1])
        #bboxes, labels = show_result(img, result, ('Fake_Pole', 'Hybrid_Pole', 'Street_Light', 'Utility_Pole'),score_thr=0.3, show = False, out_file = output_img_filename)
        bboxes = list(bboxes)
        labels = list(labels)
## ------ some more post processing to remove the duplicate bboxes, keeping the one with higher score        
        if args.post_processing:
            for idx, label in enumerate(labels):
                score = bboxes[idx][-1]
                if score < 0.5:
                    continue
                if label == 1:
                    for idx2, label2 in enumerate(labels):
                        score2 = bboxes[idx2][-1]
                        if idx2 == idx or score2 < 0.5:
                            continue
                        if bb_intersection_over_union(bboxes[idx][:-1], bboxes[idx2][:-1]) > 0.6:
                            if score - score2 > 0:
                                labels[idx2] = -1
                                bboxes[idx2] = [-1] * 5
                            else:
                                labels[idx] = -1
                                bboxes[idx] = [-1] * 5
                    #print(os.path.basename(img))
                    #print(score, score2)


## ----------- draw bboxes on images
        img_cv2 = cv2.imread(img)
        for bbox, label in zip(bboxes, labels):
            score = bbox[-1]
            if score < score_thr:
                continue
            if label == 2:
                mycol = (0, 255, 0)
            elif label == 3:
                mycol = (255, 0, 0)
            elif label == 1:
                mycol = (255, 165, 0)
            elif label == 0:
                continue

            img_cv2 = cv2.rectangle(img_cv2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), mycol, thickness=3)
            img_cv2 = cv2.putText(img_cv2, label_class[label] + ' ' + str(round(bbox[-1], 2)), (int(bbox[0] - 5), int(bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mycol, 2, cv2.LINE_AA)

        cv2.imwrite(output_img_filename, img_cv2)

        for box, label in zip(bboxes, labels):
            score = box[-1]
            if score < score_thr:
               continue
            if file_bbox:
               print(os.path.basename(img), *box, label, sep = ',', file = file_bbox)

    if file_bbox:
        file_bbox.close()



if __name__ == '__main__':
    main()
