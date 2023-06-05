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

def visualize_panoptic_segmentation(panoptic_seg, segments_info):
    frame_seg = np.zeros_like(rgb_frame)
    # frame_seg[:, :, :] = COCO_PANOPTIC_COLOR[21]  # Assume all is road
    frame_seg[:, :, :] = COCO_PANOPTIC_COLOR[40]  # Assume all is sky
    visalize_classes = [
        2,  # car
        7,  # truck
        3,  # motorcycle
        0,  # person
        40, # sky
        21, # road2
        37, # tree
    ]

    for segment_i in segments_info:
        pred_class_id = int(segment_i['id'])
        category_id = int(segment_i['category_id'])
        if category_id not in COCO_PANOPTIC_COLOR:
            print("UNKOWN:", category_id)
            # continue
            # category_id = 21 # road2
            category_id = 40 # sky
            # category_id = 148 # road
            # print('Category:', COCO_PANOPTIC_SUPERCATEGORY_COLOR[category_id], '\tName:', COCO_PANOPTIC_NAME_COLOR[category_id], '\t\tCOL:', COCO_PANOPTIC_COLOR[category_id])

        print(category_id, COCO_PANOPTIC_NAME_COLOR[category_id], COCO_PANOPTIC_COLOR[category_id])
        if category_id not in visalize_classes:
            continue
        pred_mask = panoptic_seg == pred_class_id
        frame_seg[:,:,:][pred_mask] = COCO_PANOPTIC_COLOR[category_id]
    
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

    os.chdir(os.path.expanduser("./BoostingMonocularDepth"))

    DATASET_LIST.sort()
    # DATASET_LIST = DATASET_LIST[3:]
    DATASET_LIST = [
        os.path.expanduser("~/Datasets/dataset/android/1658384924059"),
    #     os.path.expanduser("~/Datasets/dataset/android/calibration"),
    ]
    print('DATASET_LIST', DATASET_LIST)

    assert args.gen_rgb or args.gen_seg, "Nothing to do"
    assert args.gen_rgb == False, "Deprecated"


    assert (
        seg_model_cfg in SEG_MODELS
    ), "Select a segmentation model: " + "\n".join(SEG_MODELS)
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(seg_model_cfg))

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(seg_model_cfg)
    cfg.freeze()

    segmentation_predictor = DefaultPredictor(cfg)
    
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
                rgb_frame = cv2.imread(rgb_img_path)
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

                frame_seg_data = segmentation_predictor(rgb_frame)

                panoptic_seg, segments_info = frame_seg_data["panoptic_seg"]
                # sem_seg = frame_seg_data["sem_seg"]
                # instances = frame_seg_data["instances"]
                panoptic_seg = panoptic_seg.cpu().numpy()

                frame_seg = visualize_panoptic_segmentation(panoptic_seg, segments_info)
                output_segmentation_path = os.path.join(SEG_FOLDER, os.path.basename(rgb_img_path))
                # print(output_segmentation_path)
                cv2.imwrite(output_segmentation_path, frame_seg)

                # vis = Visualizer(np.zeros_like(rgb_frame[:, :, ::-1]), metadata=seg_metadata, scale=1.0)
                # print('panoptic_seg', panoptic_seg.shape, panoptic_seg.dtype)

                # print(list(frame_seg_data.keys()))
                # print('instances', instances)
                # print('sem_seg.shape', sem_seg.shape, sem_seg.dtype, sem_seg.min(), sem_seg.max())

                # cv2.imshow("rgb_frame", rgb_frame)
                # cv2.imshow("panoptic_seg", panoptic_seg)
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                #     exit()

                # seg_image_visual = visualize_semantic_segmentation(sem_seg.cpu().numpy())
                # output_segmentation_path = os.path.join(SEG_FOLDER, os.path.basename(rgb_img_path))
                # cv2.imwrite(output_segmentation_path, seg_image_visual)
                # exit()

            # data_dir = RGB_FOLDER,
            # output_dir = SEG_FOLDER,
            # TODO: Generate image segmentation data

        print("="*10)
    
    print("Done")
