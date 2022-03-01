sudo apt-get install python3.9-venv python3.9-dev build-essential swig
sudo rm -rf ./venv
python3 -m venv venv
source ./venv/bin/activate
pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip3 install -r requirements.txt
