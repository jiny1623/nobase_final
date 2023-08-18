# fastMRI_Challenge : 노베이스
2023 fastMRI final repository
## 폴더의 구조
* TODO: tree 명령어도 repository 구조 뽑기
## Training
### 1. E2E-Varnet (with MRAugment)
* ```cd root/FastMRI_challenge/e2evarnet_mraugment```로 진입합니다.
* ```python train.py --config_file experiments/paper_no_rotation_new.yaml -n e2evarnet```
  을 실행합니다.
  * random seed는 430으로 고정되어 있습니다.
  * TODO: epoch 어디까지 돌렸는지 기록
### 2. Diffusion Models
* TODO: Diffusion model training 과정
### 3. Create new h5 files using model ckpt files
* TODO: h5 file 만들어서 Data 폴더에 저장하는 과정
### 4. NAFNet
* 
