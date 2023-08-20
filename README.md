# fastMRI_Challenge : 노베이스
2023 fastMRI final repository
## 폴더의 구조
* TODO: tree 명령어도 repository 구조 뽑기
## Training
다음을 순서대로 실행합니다.
### 1. E2E-Varnet (with MRAugment)
* ```cd root/FastMRI_challenge/e2evarnet_mraugment```로 진입합니다.
* ```python train.py --config_file experiments/paper_no_rotation.yaml -n e2evarnet```
  을 실행합니다.
  * random seed는 42으로 고정되어 있습니다.
  * 42 epoch의 model을 채택했습니다. (즉 42 epoch이 완료된 후 best_model.pt를 받았습니다.)
* ```python reconstruct_and_save.py``` 을 실행합니다.
  * 각 데이터셋에 대한 reconstruction을 미리 구해 NAFNet의 training 및 evaluation 시간을 줄이기 위함입니다.
  
### 2. Baby Diffusion
* ```cd root/FastMRI_challenge/baby_diffusion```으로 진입합니다.
* ```python train.py -e 20``` 을 실행합니다.
  * random seed는 430으로 고정되어 있습니다.
* ```python reconstruct_and_save.py``` 을 실행합니다.
  * 각 데이터셋에 대한 reconstruction을 미리 구해 NAFNet의 training 및 evaluation 시간을 줄이기 위함입니다.

### 3. NAFNet
* ```cd root/FastMRI_challenge/nafnet```으로 진입합니다.
* ```python train.py -n nafnet -e 30``` 을 실행합니다.
  * random seed는 430으로 고정되어 있습니다.
 
## Reconstruction
* ```cd root/FastMRI_challenge/nafnet```로 진입합니다.
* ```python reconstruct.py -n nafnet -p [PATH_DATA]``` 을 실행합니다.

## Evaluation
* ```cd root/FastMRI_challenge```로 진입합니다.
* ```python leaderboard_eval.py -yp [PATH_MY_PATA]``` 을 실행합니다.
