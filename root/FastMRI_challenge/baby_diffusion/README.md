# Baby Diffusion
Score-based diffusion 중 Noise Conditional Score Network 모델을 훈련하는 과정입니다.\
https://github.com/yang-song/score_sde_pytorch 의 코드를 각색하여 사용했습니다.

* ```train.py```
   * train/validation을 진행하고 학습한 model의 결과를 result 폴더에 저장합니다.
   * 각 epoch마다 모델의 weight를 ```model{epoch}.pt```으로 저장합니다. 
* ```reconstruct_and_save.py```
   * ```train.py```으로 학습한 ```model20.pt```을 활용해 train, val, leaderboard dataset을 모두 reconstruction하고 그 결과를 recon_data 폴더에 저장합니다.
      * ```model20.pt```을 사용하도록 하드코딩 돼있습니다.
   * Diffusion output을 input으로 받아야 하는 NAFNet의 학습 및 평가에 걸리는 시간을 단축하기 위함입니다.
* 이외의 파일들은 2023 Baby Varnet에서 조금 수정된 파일이거나, diffusion model과 관련된 파일입니다.
