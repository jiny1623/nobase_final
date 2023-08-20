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

## 주어진 ckpt 파일을 통해서 reconstruct만 진행할 경우
* 해당 leaderboard(including hidden) data가 vessl gpu의 ```/Data/leaderboard/acc4/```, ```/Data/leaderboard/acc8/``` 안에 저장되어 있다고 간주하고 진행합니다.
* e2evarnet의 best_model.pt를 ```root/FastMRI_challenge/result/e2evarnet/checkpoints/```에 저장합니다.
* baby_diffusion의 model20.pt를 ```root/FastMRI_challenge/result/baby_diffusion/checkpoints/```에 저장합니다.
* nafnet의 best_model.pt를 ```root/FastMRI_challenge/result/nafnet/checkpoints/```에 저장합니다.

* E2E-Varnet Reconstruction
    * ```cd root/FastMRI_challenge/e2evarnet_mraugment```로 진입합니다.
    * ```python reconstruct_and_save.py``` 을 실행합니다.
* Baby Diffusion Reconstruction
    * ```cd root/FastMRI_challenge/baby_diffusion```으로 진입합니다.
    * ```python reconstruct_and_save.py``` 을 실행합니다. (약 72시간 소요)
* NAFNet Reconstuction
    * ```cd root/FastMRI_challenge/nafnet```으로 진입합니다.
    * ```python reconstruct.py``` 을 실행합니다.
* Evaluation
    * ```cd root/FastMRI_challenge```로 진입합니다.
    * ```python leaderboard_eval.py``` 을 실행합니다.