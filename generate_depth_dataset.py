import argparse
import os
import sys
import dataset_helper.dataset_iterators as dataset_iterators
from dataset_helper.dataset_constants import DATASET_LIST
from tqdm import tqdm
import cv2
import pandas

# sys.path.append(os.path.expanduser('./BoostYourOwnDepth/'))
def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

class DepthNetworks:
    MiDas=0
    LeRes=2

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="Generate depth dataset from raw dataset")
    parser.add_argument('--raw_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/dataset/android"), help='Raw dataset')
    parser.add_argument('--depth_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"), help='result dir')
    parser.add_argument('--checkpoints_dir', type=str, default="BoostYourOwnDepth/pix2pix/checkpoints", help='weights file directory')                                                                  
    
    parser.add_argument("--gen_rgb", action="store_true", help="Will generate RGB images")
    parser.add_argument("--gen_depth", action="store_true", help="Will run BoostingMonocularDepth on the RGB images")
    args = parser.parse_args()

    DEPTH_DATASET_DIR = args.depth_dataset_dir
    os.makedirs(DEPTH_DATASET_DIR, exist_ok=True)

    os.chdir(os.path.expanduser("./BoostingMonocularDepth"))

    DATASET_LIST.sort()
    DATASET_LIST = DATASET_LIST[3:]
    DATASET_LIST = [
        os.path.expanduser("~/Datasets/dataset/android/1658384924059"),
        os.path.expanduser("~/Datasets/dataset/android/calibration"),
    ]
    print('DATASET_LIST', DATASET_LIST)
    
    for android_dataset_path in DATASET_LIST:
        print("="*10)
        dataset = dataset_iterators.AndroidDatasetIterator(android_dataset_path)
        print(dataset)

        dataset_id = android_dataset_path.split('/')[-1]
        DEPTH_SUBFOLDER = os.path.join(DEPTH_DATASET_DIR, dataset_id)
        RGB_FOLDER = os.path.join(DEPTH_SUBFOLDER, "rgb_img")
        DEPTH_FOLDER = os.path.join(DEPTH_SUBFOLDER, "depth_img")
        # DEPTH_NPY_FOLDER = os.path.join(DEPTH_SUBFOLDER, "depth_npy")
        CSV_PATH = os.path.join(DEPTH_SUBFOLDER, dataset_id+".csv")

        os.makedirs(DEPTH_SUBFOLDER, exist_ok=True)
        os.makedirs(RGB_FOLDER, exist_ok=True)
        os.makedirs(DEPTH_FOLDER, exist_ok=True)
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

        if args.gen_depth:
            print('Generating Depth data: ', DEPTH_FOLDER)
            command = "python3 run.py --Final --data_dir {data_dir} --output_dir  {output_dir} --depthNet {depthNet}".format(
                data_dir = RGB_FOLDER,
                output_dir = DEPTH_FOLDER,
                depthNet = DepthNetworks.MiDas
            )

            do_system(command)

        print("="*10)
    
    print("Done")


# python3 ./BoostYourOwnDepth/boost_depth.py --data_dir ./BoostYourOwnDepth/input/ --output_dir ./BoostYourOwnDepth/output --checkpoints_dir ./BoostYourOwnDepth/pix2pix/checkpoints --colorize_results