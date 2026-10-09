# Geometry-Dash-AI
Geometry Dash에서 강화 학습을 통한 장애물 통과 여부를 시험하는 프로그램입니다.

필요 모듈

torch
opencv-python
mss
pynput
numpy

사용:
  1) 게임 설정에서 "Show Progress Bar(진행률 바 표시)" 켜기, 창모드로 실행(예: 1280x720), 연습모드(Practice) 끄기
  2) python gd_real.py calibrate     # 게임 화면 영역, 진행률 바 영역을 마우스로 선택 (gd_config.json 저장)
  3) python gd_real.py test          # 전처리 화면과 진행률 인식(빨간 선)이 맞는지 확인 (q로 종료)

  --- 사람 플레이 학습 (Behavioral Cloning) ---
  4) python gd_real.py record        # 사람이 플레이하는 동안 화면+키 입력을 녹화 (F8로 중지, demos.npz 저장)
     * 여러 번 녹화하면 기존 demos.npz에 이어 붙입니다.
  5) python gd_real.py bc            # 녹화한 데모로 Behavioral Cloning 학습 (사람 행동을 모방)
  6) python gd_real.py play --load gd_bc.pt   # BC로 학습된 모델로 스스로 플레이

  --- 기존 RL (자기 플레이로 DQN 학습) ---
  7) python gd_real.py train         # 5초 카운트다운 동안 게임 창을 클릭해 포커스 -> 학습 시작
  8) python gd_real.py play --load gd_real.pt    # 학습된 모델로 플레이만 (학습 X)
  * 학습 중지: F8 키 (모델 자동 저장).  이어서 학습: python gd_real.py train --load gd_real.pt

  팁: BC로 기본기를 익힌 뒤 train --load gd_bc.pt 로 RL 파인튜닝하면 더 잘 됩니다.
