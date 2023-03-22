# Install gdown
python3 -m pip install gdown

# Downloading merge model weights
gdown https://drive.google.com/u/0/uc?id=1cU2y-kMbt0Sf00Ns4CN2oO9qPJ8BensP&export=download
wait $!

mkdir -p ./BoostYourOwnDepth/pix2pix/checkpoints/mergemodel/
mv ./latest_net_G.pth ./BoostYourOwnDepth/pix2pix/checkpoints/mergemodel/

