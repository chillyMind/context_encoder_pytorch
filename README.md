## Context Encoders: Feature Learning by Inpainting

This is the Pytorch implement of [CVPR 2016 paper on Context Encoders](http://cs.berkeley.edu/~pathak/context_encoder/)

![corrupted image size](https://github.com/chillyMind/context_encoder_pytorch/blob/master/blob/cropped_samples.png){:height="500px" width="500px"}
![result image size](https://github.com/chillyMind/context_encoder_pytorch/blob/master/blob/recon_center_samples.png){:height="500px" width="500px"}
### 1) Semantic Inpainting Demo

1. Install PyTorch http://pytorch.org/

2. Clone the repository
  ```Shell
  git clone https://github.com/BoyuanJiang/context_encoder_pytorch.git
  ```
3. Demo

    Download pre-trained model on Paris Streetview from
    [Google Drive](https://drive.google.com/open?id=0B6oeoQaX0xmzS0RXXzNYZkZ3ZUk) OR [BaiduNetdisk](https://pan.baidu.com/s/1hsLzJPq)
    ```Shell
    cp netG_streetview.pth context_encoder_pytorch/model/
    cd context_encoder_pytorch/model/
    # Inpainting a batch iamges
    python test.py --netG model/netG_streetview.pth --dataroot dataset/val --batchSize 100
    # Inpainting one image 
    python test_one.py --netG model/netG_streetview.pth --test_image result/test/cropped/065_im.png
    ```

### 2) Train on your own dataset
1. Build dataset

    Put your images under dataset/train,all images should under subdirectory

    dataset/train/subdirectory1/some_images
    
    dataset/train/subdirectory2/some_images

    ...
    
    **Note**:For Google Policy,Paris StreetView Dataset is not public data,for research using please contact with [pathak22](https://github.com/pathak22).
    You can also use [The Paris Dataset](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) to train your model

2. Train (alpha map cropped isnt implemented in this method)
```Shell
python train.py --cuda --wtl2 0.999 --niter 200
```
with jittering
```Shell
python train.py [options ...] --jittering
```

3. Test

    This step is similar to [Semantic Inpainting Demo](#1-semantic-inpainting-demo)

    
### 3) Train on your own dataset with alpha map cropped (by ipython)
1. Build dataset
    Put your images under dataset/train,all images should under subdirectory

    dataset/train/subdirectory1/some_images
    
    dataset/train/subdirectory2/some_images
    ...
    
    dataset/pngdata/subdirectory1/some_images
    
    dataset/pngdata/subdirectory21/some_images
    ...
    
    ### Alpha map img constraint
    The images are cosist of black and white pixels.
    black for being not cropped, white for being cropped.
    ex)
    ![pngsample](https://github.com/chillyMind/context_encoder_pytorch/blob/master/blob/png%20sample.jpg |=100x100)

2. Train (noneSquare_train_alphamap.ipynb)
    When you in first trainning, set and remvoe below annotations 
```Shell
class opt():
  def __init__(self):
  ...
  ...
        self.netG=''
        self.netD=''
        #self.netG='...'
        #self.netG='...'
  ...
```

    When you in first trainning, set and remove below annotations
```Shell
class opt():
  def __init__(self):
  ...
  ...

        #self.netG=''
        #self.netD=''
        self.netG='...'
        self.netG='...'
  ...
```

3. Test (noneSquare_test_singleimg.ipynb)
```Shell
class opt():
  def __init__(self):
  ...
  self.testimg = 'IMG_TO_TEST'
  ...
```
![inputS](https://github.com/chillyMind/context_encoder_pytorch/blob/master/blob/single_test_image(cropped)_rabbit_cropped.png)
![resultS](https://github.com/chillyMind/context_encoder_pytorch/blob/master/blob/single_test_image(mask)__rabbit_cropped.png)
