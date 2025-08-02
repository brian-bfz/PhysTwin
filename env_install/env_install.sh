conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender
pip install h5py

conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma

# Install the env for realsense camera
pip install Cython
pip install atomics
pip install pynput

pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

pip install gsplat==1.4.0
pip install kornia
cd gaussian_splatting/
pip install submodules/simple-knn/
cd ..
