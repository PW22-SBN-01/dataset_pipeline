# Install gdown
python3 -m pip install gdown

# Downloading merge model weights
gdown https://drive.google.com/u/0/uc?id=1cU2y-kMbt0Sf00Ns4CN2oO9qPJ8BensP&export=download
wait $!

mkdir -p ./BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/
mv latest_net_G.pth ./BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/

# Downloading Midas weights
wget https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt
mv midas_v21-f6b98070.pt ./BoostingMonocularDepth/midas/model.pt

# Downloading LeRes weights
wget https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download
mv download ./BoostingMonocularDepth/res101.pth