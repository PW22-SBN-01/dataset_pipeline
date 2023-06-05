import argparse
import os
import sys
import dataset_helper.dataset_iterators as dataset_iterators
from dataset_helper.dataset_constants import DATASET_LIST
from tqdm import tqdm
import cv2
import pandas
import glob
import numpy as np

# sys.path.append(os.path.expanduser('./BoostYourOwnDepth/'))
def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

import numpy as np
from constants import COCO_PANOPTIC_COLOR, SEG_MODELS, COCO_PANOPTIC_NAME_COLOR

def visualize_semantic_segmentation(results, output, rgb_frame):
    # ['boxes', 'class_ids', 'class_names', 'object_counts', 'scores', 'masks', 'extracted_objects']
    pred_classes = results["class_ids"]
    pred_masks = results["masks"]             # Getting Masks for all classes
    pred_masks = pred_masks.transpose((2, 0, 1))   # Converting AxBxC into CxAxB to get each mask in masks into required shape of img, AxB

    # print(list(results.keys()))
    # exit()
    
    # pred_masks = instances.pred_masks.cpu().numpy()
    # pred_classes = instances.pred_classes.cpu().numpy()
    
    # print('pred_masks', pred_masks.shape, pred_masks.dtype)
    # print('pred_classes', pred_classes.shape, pred_classes.dtype)
    
    frame_seg = np.zeros_like(rgb_frame)
    # frame_seg[:, :, :] = COCO_PANOPTIC_COLOR[21]  # Assume all is road
    # frame_seg[:, :, :] = COCO_PANOPTIC_COLOR[40]  # Assume all is sky
    frame_seg[:, :, :] = 0.0  # Assume all is void
    visalize_classes = [
        2,  # car
        7,  # truck
        3,  # motorcycle
        5,  # bus
        0,  # person
    ]
    N = min(len(pred_masks), len(pred_classes))
    for i in range(N):
        pred_mask_i = pred_masks[i]
        pred_class_i = pred_classes[i]


        if pred_class_i not in COCO_PANOPTIC_COLOR:
            # print("UNKOWN:", pred_class_i)
            continue

        # print(pred_class_i, COCO_PANOPTIC_NAME_COLOR[pred_class_i], COCO_PANOPTIC_COLOR[pred_class_i])

        if pred_class_i not in visalize_classes:
            continue

        if pred_class_i in [2, 7, 3, 5]:
            pred_class_i = 2 # Vehicle

        color = COCO_PANOPTIC_COLOR[pred_class_i]
        frame_seg[pred_mask_i] = color


    return frame_seg


seg_model_cfg = SEG_MODELS[0]

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="Generate depth dataset from raw dataset")
    parser.add_argument('--raw_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/dataset/android"), help='Raw dataset')
    parser.add_argument('--seg_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"), help='result dir')
    
    parser.add_argument("--gen_rgb", action="store_true", help="Will generate RGB images")
    parser.add_argument("--gen_seg", action="store_true", help="Will run BoostingMonocularDepth on the RGB images")
    args = parser.parse_args()

    SEG_DATASET_DIR = args.seg_dataset_dir
    os.makedirs(SEG_DATASET_DIR, exist_ok=True)

    DATASET_LIST.sort()
    # DATASET_LIST = DATASET_LIST[3:]
    # DATASET_LIST = [
    #     os.path.expanduser("~/Datasets/dataset/android/1658384924059"),
    #     # os.path.expanduser("~/Datasets/dataset/android/calibration"),
    # ]
    print('DATASET_LIST', DATASET_LIST)

    assert args.gen_rgb or args.gen_seg, "Nothing to do"
    assert args.gen_rgb == False, "Deprecated"


    import pixellib
    from pixellib.torchbackend.instance import instanceSegmentation

    segmentation_predictor = instanceSegmentation()
    # segmentation_predictor.load_model("weights/pointrend_resnet50.pkl")
    segmentation_predictor.load_model("weights/pointrend_resnet101.pkl", network_backbone = "resnet101")
    
    for android_dataset_path in DATASET_LIST:
        print("="*10)
        dataset = dataset_iterators.AndroidDatasetIterator(android_dataset_path)
        print(dataset)

        dataset_id = android_dataset_path.split('/')[-1]
        SEG_SUBFOLDER = os.path.join(SEG_DATASET_DIR, dataset_id)
        RGB_FOLDER = os.path.join(SEG_SUBFOLDER, "rgb_img")
        SEG_FOLDER = os.path.join(SEG_SUBFOLDER, "seg_img")
        # DEPTH_NPY_FOLDER = os.path.join(SEG_SUBFOLDER, "depth_npy")
        CSV_PATH = os.path.join(SEG_SUBFOLDER, dataset_id+".csv")

        os.makedirs(SEG_SUBFOLDER, exist_ok=True)
        os.makedirs(RGB_FOLDER, exist_ok=True)
        os.makedirs(SEG_FOLDER, exist_ok=True)
        # os.makedirs(DEPTH_NPY_FOLDER, exist_ok=True)

        print('dataset_id', dataset_id)

        if args.gen_rgb and not os.path.isfile(CSV_PATH):
            print('Generating RGB data: ', RGB_FOLDER)

            csv_data = {
                'Timestamp'    : [],
                'Longitude'    : [],
                'Latitude'     : [],
                'RotationV X': [],
                'RotationV Y': [],
                'RotationV Z': [],
                'RotationV W': [],
                'RotationV Acc': [],
                'linear_acc_x' : [],
                'linear_acc_y' : [],
                'linear_acc_z' : [],
                'heading'      : [],
                'speed'        : [],
            }

            # for frame in tqdm(dataset):
            # for timestmap in tqdm(
            #     range(int(dataset.start_time_csv), int(dataset.end_time_csv), int(1000.0/dataset.fps))
            # ): # ms
                # frame = dataset.get_item_by_timestamp(timestmap)
            # for index in tqdm(range(0, len(dataset)-50, 2)): # 10 FPS
            for index in tqdm(range(0, len(dataset)-50, 20)): # 1 FPS
                try:
                    frame = dataset[index]
                    frame_csv, frame_img = frame
                    for key in frame_csv.keys():
                        csv_data[key] += [frame_csv[key]]

                    frame_ts = str(int(frame_csv['Timestamp']))
                    img_path = os.path.join(RGB_FOLDER, frame_ts+'.png')
                    if not os.path.isfile(img_path):
                        cv2.imwrite(img_path, frame_img)
                except Exception as ex:
                    print(ex)
            
            csv_df = pandas.DataFrame(csv_data)
            csv_df.to_csv(CSV_PATH)

        if args.gen_seg:
            print('Generating Segmentation data: ', SEG_FOLDER)


            # iterate over images in RGB_FOLDER
            rgb_img_list = sorted(glob.glob(os.path.join(RGB_FOLDER, "*.png")))
            for rgb_img_path in tqdm(rgb_img_list):
                output_segmentation_path = os.path.join(SEG_FOLDER, os.path.basename(rgb_img_path))
                if os.path.isfile(output_segmentation_path):
                    continue
                rgb_frame = cv2.imread(rgb_img_path)
                # rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

                results, output = segmentation_predictor.segmentImage(rgb_img_path)

                frame_seg = visualize_semantic_segmentation(results, output, rgb_frame)
                
                # print(output_segmentation_path)
                cv2.imwrite(output_segmentation_path, frame_seg)

        print("="*10)
    
    print("Done")
