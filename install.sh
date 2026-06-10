# conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install cython
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng

pip install cython==0.29.36
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"

cd models/dqdetr/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..

