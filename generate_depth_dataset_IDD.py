import argparse
import os
import sys
import dataset_helper.dataset_iterators as dataset_iterators
from dataset_helper.dataset_constants import DATASET_LIST
from tqdm import tqdm
import glob

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
    parser = argparse.ArgumentParser(description="Generate depth dataset for IDD")
    parser.add_argument('--raw_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/IDD_Segmentation/leftImg8bit"), help='Raw dataset')
    parser.add_argument('--depth_dataset_dir', type=str, default=os.path.expanduser("~/Datasets/IDD_Segmentation/depth"), help='result dir')
    parser.add_argument('--checkpoints_dir', type=str, default="BoostYourOwnDepth/pix2pix/checkpoints", help='weights file directory')                                                                  
    
    parser.add_argument("--gen_rgb", action="store_true", help="Will generate RGB images")
    parser.add_argument("--gen_depth", action="store_true", help="Will run BoostingMonocularDepth on the RGB images")
    args = parser.parse_args()

    DEPTH_DATASET_DIR = args.depth_dataset_dir
    os.makedirs(DEPTH_DATASET_DIR, exist_ok=True)

    os.chdir(os.path.expanduser("./BoostingMonocularDepth"))

    assert args.gen_rgb == False, "Not implemented"

    DATASET_LIST = glob.glob(os.path.join(args.raw_dataset_dir, "*", "*"))
    DATASET_LIST.sort()

    print('DATASET_LIST', DATASET_LIST)

    for idd_dataset_path in tqdm(DATASET_LIST):

        if args.gen_depth:
            depth_dataset_path = idd_dataset_path.replace(args.raw_dataset_dir, DEPTH_DATASET_DIR)
            os.makedirs(depth_dataset_path, exist_ok=True)

            # print('Generating Depth data: ', depth_dataset_path)
            command = "python3 run.py --Final --data_dir {data_dir} --output_dir  {output_dir} --depthNet {depthNet}".format(
                data_dir = idd_dataset_path,
                output_dir = depth_dataset_path,
                depthNet = DepthNetworks.MiDas
            )

            # print(command)

            # do_system(command)
    
    print("Done")

