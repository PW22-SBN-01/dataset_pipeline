import argparse
import os
import sys
import dataset_helper.dataset_iterators as dataset_iterators
from dataset_helper.dataset_constants import DATASET_LIST
from tqdm import tqdm
import cv2
import pandas

# sys.path.append(os.path.expanduser('./BoostYourOwnTrajectory/'))
def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="Generate trajectory dataset from raw dataset")
    parser.add_argument('--raw_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/dataset/android"), help='Raw dataset')
    parser.add_argument('--trajectory_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"), help='result dir')
    
    # parser.add_argument("--gen_rgb", action="store_true", help="Will generate RGB images")
    parser.add_argument("--gen_traj", action="store_true", help="Will run BoostingMonocularTrajectory on the RGB images")
    args = parser.parse_args()

    TRAJECTORY_DATASET_DIR = args.trajectory_dataset_dir
    os.makedirs(TRAJECTORY_DATASET_DIR, exist_ok=True)


    # DATASET_LIST.sort()
    # DATASET_LIST = DATASET_LIST[3:]
    DATASET_LIST = [
        os.path.expanduser("~/Datasets/dataset/android/1658384707877"),
        # os.path.expanduser("~/Datasets/dataset/android/1658384924059"),
        # os.path.expanduser("~/Datasets/dataset/android/calibration"),
    ]
    print('DATASET_LIST', DATASET_LIST)
    
    for android_dataset_path in DATASET_LIST:
        print("="*10)
        dataset = dataset_iterators.AndroidDatasetIterator(
            android_dataset_path,
            invalidate_cache=False,
		    compute_trajectory=True,
        )
        print(dataset)

        dataset_id = android_dataset_path.split('/')[-1]
        TRAJECTORY_SUBFOLDER = os.path.join(TRAJECTORY_DATASET_DIR, dataset_id)
        RGB_FOLDER = os.path.join(TRAJECTORY_SUBFOLDER, "rgb_img")
        # TRAJECTORY_FOLDER = os.path.join(TRAJECTORY_SUBFOLDER, "trajectory_img")
        # TRAJECTORY_NPY_FOLDER = os.path.join(TRAJECTORY_SUBFOLDER, "trajectory_npy")
        CSV_PATH = os.path.join(TRAJECTORY_SUBFOLDER, dataset_id+"_traj.csv")

        os.makedirs(TRAJECTORY_SUBFOLDER, exist_ok=True)
        os.makedirs(RGB_FOLDER, exist_ok=True)
        # os.makedirs(TRAJECTORY_FOLDER, exist_ok=True)
        # os.makedirs(TRAJECTORY_NPY_FOLDER, exist_ok=True)

        print('dataset_id', dataset_id)

        # if args.gen_traj and not os.path.isfile(CSV_PATH):
        if args.gen_traj:
            print('Generating Trajectory data: ', RGB_FOLDER)

            csv_data = {
                'Timestamp'    : [],
                
                'x':[], 'y':[], 'z': [], 'rot': []
            }

            # for frame in tqdm(dataset):
            # for timestmap in tqdm(
            #     range(int(dataset.start_time_csv), int(dataset.end_time_csv), int(1000.0/dataset.fps))
            # ): # ms
                # frame = dataset.get_item_by_timestamp(timestmap)
            # for index in tqdm(range(0, len(dataset)-50, 2)): # 10 FPS
            # for index in tqdm(range(0, len(dataset)-50, 20)): # 1 FPS
            for index in tqdm(range(0, len(dataset))):
                try:
                    frame = dataset[index]
                    traj = dataset.trajectory.iloc[index]
                    frame_csv, frame_img = frame
                    
                    
                    csv_data['Timestamp'] += [frame_csv['Timestamp']]
                    csv_data['x'] += [traj['x']]
                    csv_data['y'] += [traj['y']]
                    csv_data['z'] += [traj['z']]
                    csv_data['rot'] += [traj['rot']]

                    frame_ts = str(int(frame_csv['Timestamp']))
                    # img_path = os.path.join(RGB_FOLDER, frame_ts+'.png')
                    # if not os.path.isfile(img_path):
                    #     cv2.imwrite(img_path, frame_img)
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    print(ex)
                    exit(1)
            
            csv_df = pandas.DataFrame(csv_data)
            csv_df.to_csv(CSV_PATH)

            
        print("="*10)
    
    print("Done")


