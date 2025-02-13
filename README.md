# Edge-constrained Deep Unfolding Network for Image Resolution
Our project dedicated to use edge prior as constraint for better guiding image reconstruction process, which could enchance the final reconstruction effects.
## Folder Structure of Project
```
├─main.py

├─option.py

├─requirements.txt

├─README.md

├─trainer.py

├─utility.py


├─loss

│  ├─_init_.py

│  ├─adversarial.py

│  ├─vgg.py


├─models

│  ├─_init_.py

│  ├─denoisingModule.py

│  ├─ecdun.py

│  ├─edgefeatureextractionModule.py

│  ├─edgemap.py

│  ├─intermediatevariableupdateModule.py

│  ├─residualprojectionModule.py

│  ├─textureReconstructionModule.py

│  ├─variableguidereconstructionModule.py


├─mydata

│  ├─_init_.py

│  ├─benchmark.py

│  ├─common.py

│  ├─div2k.py

│  ├─myDataLoader.py

│  ├─srdata.py
```

## Package Dependencies
This project is built with Pytorch 1.0.1, Python3.7,CUDA10.0. For package dependencies, you can install them by:

`pip install -r requirements.txt`

## Content of requirements.txt

```
torch~=1.0.1
matplotlib~=3.0.3
numpy~=1.16.2
scipy~=1.2.1
scikit-image~=0.14.2
imageio~=2.5.0
tqdm~=4.31.1
torchvision~=0.2.0
scikit-learn~=0.20.3
```

## Training
To train base model on DIV2K, we use 3090 GPU and run for 300 epochs:
```
python main.py --data_train DIV2K --epochs 300 --save_results SAVE_RESULTS
```
If you want to train model with different scales, you can add additional parameter like (default scale is x2):
```
python main.py --data_train DIV2K --epochs 300 --save_results SAVE_RESULTS --scale 4
```
