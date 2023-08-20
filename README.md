# fastMRI_Challenge : 노베이스
2023 fastMRI final repository
## Training
다음을 순서대로 실행합니다.
### 0. Requirements
* requirements 설치는 초기화된 vessl gpu에서 ```pip install -r requirements.txt```를 실행하면 완료됩니다.
### 1. E2E-Varnet (with MRAugment)
* ```cd root/FastMRI_challenge/e2evarnet_mraugment```로 진입합니다.
* ```python train.py --config_file experiments/paper_no_rotation.yaml```
  을 실행합니다.
  * random seed는 420으로 고정되어 있습니다.
  * 42 epoch의 model을 채택했습니다. (즉 42 epoch이 완료된 후 best_model.pt를 받았습니다.)
* ```python reconstruct_and_save.py``` 을 실행합니다.
  * 각 데이터셋에 대한 reconstruction을 미리 구해 NAFNet의 training 및 evaluation 시간을 줄이기 위함입니다.
  
### 2. Baby Diffusion
* ```cd root/FastMRI_challenge/baby_diffusion```으로 진입합니다.
* ```python train.py -e 20``` 을 실행합니다.
  * random seed는 430으로 고정되어 있습니다.
* ```python reconstruct_and_save.py``` 을 실행합니다.
  * diffusion reconstruct 과정은 약 72시간이 소요됩니다.
  * 각 데이터셋에 대한 reconstruction을 미리 구해 NAFNet의 training 및 evaluation 시간을 줄이기 위함입니다.

### 3. NAFNet
* ```cd root/FastMRI_challenge/nafnet```으로 진입합니다.
* ```python train.py -e 30``` 을 실행합니다.
  * random seed는 430으로 고정되어 있습니다.
  * 11 epoch의 model을 채택했습니다. (즉 11 epoch이 완료된 후 best_model.pt를 받았습니다.)
  
## Reconstruction
* ```cd root/FastMRI_challenge/nafnet```로 진입합니다.
* ```python reconstruct.py``` 을 실행합니다.

## Evaluation
* ```cd root/FastMRI_challenge```로 진입합니다.
* ```python leaderboard_eval.py``` 을 실행합니다.
