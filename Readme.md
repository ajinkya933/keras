I am Following this Tutorial:
https://www.analyticsvidhya.com/blog/2019/05/understanding-visualizing-neural-networks/

1) Create: keras virtual enviornment using: 
```
conda create -n tensorflow_p36 python=3.6 anaconda
```

2) Install keras-gpu on this virtual enviornment using: 
```
conda install -c anaconda keras-gpu 
```
3) uninstall Tensorflow 2.0, Tensorflow 2.0 comes by default installed on keras-gpu virtual enviornment.
```
pip uninstall tensorflow
```
4) Install tensorflow 1.14-gpu
```
pip install tensorflow-gpu==1.14
```

Install these libraries:
```
sudo pip install keras-vis
pip install git+https://github.com/raghakot/keras-vis.git
```
Run ```demo.py```
Note:To change the image from "indian elephant to something else" please change the value of filter_indices from 384 to 380
