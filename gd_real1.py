"""
실제 Geometry Dash 게임에 화면 캡처로 붙여서 학습하는 Double DQN + 인간 데모 학습 에이전트

에이전트가 받는 정보:  게임 화면(흑백 84x48, 최근 4프레임 + 프레임 차분 3채널)  +  보상
  - 입력 채널 = 4(원본 프레임) + 3(연속 프레임 차분) = 7채널 → 움직임/속도 정보 보강
  - 보상 = 화면 상단 '진행률 바'가 채워진 만큼(앞으로 간 만큼) + 죽으면 -1
  - 장애물이 뭔지, 언제 점프해야 하는지는 전혀 알려주지 않음
게임 메모리/내부값은 읽지 않고, 오직 화면 캡처 + 스페이스 키 입력만 사용합니다.

※ 프레임 차분 추가 후에는 기존 demos.npz / 모델을 다시 녹화·학습해야 합니다 (채널 수 변경).

설치:   pip install torch opencv-python mss pynput numpy
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
"""
import os, sys, json, time, random, argparse, threading, collections
import numpy as np

try:
    import cv2, mss, torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pynput import keyboard
except ImportError as e:
    sys.exit(f"필요한 패키지가 없습니다: {e}\n  pip install torch opencv-python mss pynput numpy")

CFG_PATH = "gd_config.json"
DEMO_PATH = "demos.npz"
IMG_W, IMG_H, STACK = 84, 48, 4
# 관측 = STACK개 원본 프레임 + (STACK-1)개 프레임 차분 → 움직임 정보
CHANNELS = STACK + (STACK - 1)  # 4 + 3 = 7


def make_obs(frames):
    """그레이스케일 프레임 리스트/배열 (STACK, H, W) → (CHANNELS, H, W) uint8
    차분 = (현재 - 이전) + 128 로 매핑 (128=변화없음, 밝을수록 밝아짐, 어두울수록 어두워짐)
    """
    f = np.stack(frames).astype(np.uint8) if not isinstance(frames, np.ndarray) else frames.astype(np.uint8)
    if f.shape[0] != STACK:
        raise ValueError(f"expected {STACK} frames, got {f.shape[0]}")
    # signed diff, then shift to 0..255
    d = f[1:].astype(np.int16) - f[:-1].astype(np.int16)
    d = np.clip(d + 128, 0, 255).astype(np.uint8)
    return np.concatenate([f, d], axis=0)  # (STACK + STACK-1, H, W)


# ----------------------------------------------------------------------------
# 0) 영역 설정 (마우스로 게임 화면 / 진행률 바 영역 선택)
# ----------------------------------------------------------------------------
def calibrate(monitor):
    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        shot = np.asarray(sct.grab(mon))[:, :, :3].copy()
    sc = min(1.0, 1500 / shot.shape[1], 850 / shot.shape[0])
    small = cv2.resize(shot, None, fx=sc, fy=sc)
    print("[1/2] 게임 플레이 화면 전체를 드래그로 선택하고 Enter")
    r1 = cv2.selectROI("1) game area  (Enter)", small, False)
    cv2.destroyAllWindows()
    if r1[2] == 0:
        sys.exit("영역이 선택되지 않았습니다.")
    x, y, w, h = [int(v / sc) for v in r1]                    # 모니터 기준 원본 좌표
    # 진행률 바는 상단 중앙에 있으므로, 그 부분만 원본 해상도로 잘라 확대해서 정밀하게 선택
    ox, oy = x + int(w * 0.2), y
    strip = shot[oy:oy + max(h // 5, 30), ox:x + int(w * 0.8)]
    z = 1500 / strip.shape[1]
    zoom = cv2.resize(strip, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    print("[2/2] 확대된 상단 화면에서 진행률 바를 선택하고 Enter\n"
          "      - 가로: 바(트랙)의 왼쪽 끝 ~ 오른쪽 끝에 정확히 맞추기 (0%~100% 전체)\n"
          "      - 세로: 바 안쪽만 얇게 (테두리/배경 제외)")
    r2 = cv2.selectROI("2) progress bar  (Enter)", zoom, False)
    cv2.destroyAllWindows()
    if r2[2] == 0:
        sys.exit("영역이 선택되지 않았습니다.")
    bar = [ox + int(r2[0] / z), oy + int(r2[1] / z), max(int(r2[2] / z), 4), max(int(r2[3] / z), 1)]
    json.dump({"monitor": monitor, "game": [x, y, w, h], "bar": bar}, open(CFG_PATH, "w"))
    print(f"저장 완료: {CFG_PATH}   (다음: python gd_real.py test)")


# ----------------------------------------------------------------------------
# 1) 화면 캡처 + 전처리 + 진행률 읽기
# ----------------------------------------------------------------------------
class Screen:
    def __init__(self, crop=(0.0, 1.0, 0.0, 1.0)):
        self.crop = crop
        cfg = json.load(open(CFG_PATH))
        self.sct = mss.mss()
        mon = self.sct.monitors[cfg["monitor"]]
        self.g, self.b = cfg["game"], cfg["bar"]
        x0, y0 = min(self.g[0], self.b[0]), min(self.g[1], self.b[1])
        x1 = max(self.g[0] + self.g[2], self.b[0] + self.b[2])
        y1 = max(self.g[1] + self.g[3], self.b[1] + self.b[3])
        self.o = (x0, y0)
        self.region = {"left": mon["left"] + x0, "top": mon["top"] + y0, "width": x1 - x0, "height": y1 - y0}
        self.last_prog, self.raw, self.split, self.dbg_bar = 0.0, 0.0, 0, None
        self.fill_col = self.track_col = None
        self.buf = collections.deque(maxlen=3)            # 3프레임 중앙값으로 순간 오류 제거
        self.min_step = 0.4 / max(self.b[2], 1)          # 진행률이 '증가했다'고 보는 최소 변화 (바 0.4픽셀)

    def read(self):
        img = np.asarray(self.sct.grab(self.region))[:, :, :3]
        ox, oy = self.o
        gx, gy, gw, gh = self.g; bx, by, bw, bh = self.b
        game = img[gy - oy:gy - oy + gh, gx - ox:gx - ox + gw]
        bar = img[by - oy:by - oy + bh, bx - ox:bx - ox + bw]
        x0, x1, y0, y1 = self.crop                       # 플레이어와 앞쪽 장애물 영역만 사용 (해상도 효율 UP)
        gh_, gw_ = game.shape[:2]
        game = game[int(gh_ * y0):int(gh_ * y1), int(gw_ * x0):int(gw_ * x1)]
        gray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        self.dbg_bar = bar.copy()
        self.raw = self._progress(bar)
        self.buf.append(self.raw)
        self.last_prog = float(np.median(self.buf))
        return small, self.last_prog

    def reset(self):
        self.buf.clear()

    def _progress(self, bar):
        """바 안에서 '왼쪽 채움색 | 오른쪽 빈 트랙색' 경계 위치를 찾는다.
        밝기 임계값 대신 색 분산을 최소화하는 분할점을 찾으므로 바 색/테마와 무관하게 동작."""
        h, w = bar.shape[:2]
        if w < 4:
            return self.last_prog
        mid = bar[int(h * 0.25):max(int(h * 0.75), int(h * 0.25) + 1)]
        cols = np.median(mid.astype(np.float64), axis=0)            # (w,3) 열별 대표 색
        c1, c2 = np.cumsum(cols, 0), np.cumsum(cols ** 2, 0)
        t1, t2 = c1[-1], c2[-1]
        n = np.arange(1, w)
        ls, lq = c1[n - 1], c2[n - 1]
        rs, rq = t1 - ls, t2 - lq
        cost = ((lq - ls ** 2 / n[:, None]) + (rq - rs ** 2 / (w - n)[:, None])).sum(1)
        i = int(cost.argmin())
        ml, mr = ls[i] / n[i], rs[i] / (w - n[i])
        sep = float(np.abs(ml - mr).max())
        total = float((t2 - t1 ** 2 / w).sum())
        if sep >= 35 and cost[i] < 0.5 * total:                      # 뚜렷한 경계 발견
            self.fill_col, self.track_col = ml, mr
            self.split = int(n[i])
            return self.split / w
        m = t1 / w                                                    # 단색: 완전히 비었거나 가득 참
        full = self.fill_col is not None and np.abs(m - self.fill_col).max() < np.abs(m - self.track_col).max()
        self.split = w if full else 0
        return 1.0 if full else 0.0


class Pad:
    """스페이스 키를 누르고 있는 상태(=점프 입력)를 관리"""
    def __init__(self):
        self.kb, self.held = keyboard.Controller(), False

    def set(self, a):
        if a == 1 and not self.held:
            self.kb.press(keyboard.Key.space); self.held = True
        elif a == 0 and self.held:
            self.kb.release(keyboard.Key.space); self.held = False


def test(args):
    scr = Screen(args.crop)
    frames = collections.deque(maxlen=STACK)
    print("q 키로 종료. 진행률이 게임 진행에 맞춰 0~1로 올라가는지 확인하세요.")
    print("왼쪽: 에이전트가 보는 최신 프레임 / 오른쪽: 프레임 차분(움직임)")
    while True:
        frame, prog = scr.read()
        frames.append(frame)
        if len(frames) < STACK:
            continue
        obs = make_obs(frames)
        # 최신 원본 프레임
        view = cv2.cvtColor(cv2.resize(obs[STACK - 1], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR)
        cv2.putText(view, f"progress {prog*100:5.1f}%", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # 가장 최근 차분 (마지막 채널)
        diff = obs[-1]
        dview = cv2.cvtColor(cv2.resize(diff, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_GRAY2BGR)
        cv2.putText(dview, "frame diff (128=no change)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        both = np.hstack([view, dview])
        cv2.imshow("agent view | frame diff  (q=quit)", both)
        bar = scr.dbg_bar
        bw = 1000
        bv = cv2.resize(bar, (bw, max(bar.shape[0] * 6, 60)), interpolation=cv2.INTER_NEAREST)
        sx = int(scr.split / bar.shape[1] * bw)
        cv2.line(bv, (sx, 0), (sx, bv.shape[0]), (0, 0, 255), 2)          # 빨간 선 = 인식된 진행 위치
        cv2.putText(bv, f"raw {scr.raw*100:5.1f}%  filtered {prog*100:5.1f}%", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow("progress bar crop (red line = detected)", bv)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break


# ----------------------------------------------------------------------------
# 2) 신경망 (CNN) + 경험 재생
# ----------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, CHANNELS, IMG_H, IMG_W)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.fc(self.conv(x.float() / 255.0))


class Replay:
    def __init__(self, n, nstep):
        self.n, self.nstep, self.ptr, self.size, self.lock = n, nstep, 0, 0, threading.Lock()
        self.obs = np.zeros((n, CHANNELS, IMG_H, IMG_W), np.uint8)
        self.act = np.zeros(n, np.int64); self.rew = np.zeros(n, np.float32)
        self.done = np.zeros(n, np.float32); self.near = np.zeros(n, bool)

    def add_episode(self, obs, act, rew, done, near):    # rew: n-step 누적 보상, 다음 상태 = nstep 뒤 관측
        with self.lock:
            for o, a, r, d, nr in zip(obs, act, rew, done, near):
                p = self.ptr
                self.obs[p], self.act[p], self.rew[p], self.done[p], self.near[p] = o, a, r, d, nr
                self.ptr, self.size = (p + 1) % self.n, min(self.size + 1, self.n)

    def sample(self, b, near_frac):
        with self.lock:
            i = np.random.randint(0, self.size, b)
            pool = np.flatnonzero(self.near[:self.size])
            k = int(b * near_frac)
            if k > 0 and len(pool) > 0:                   # 일부는 '죽기 직전 2초' 장면에서 뽑아 학습 신호 강화
                i[:k] = pool[np.random.randint(0, len(pool), k)]
            return self.obs[i], self.act[i], self.rew[i], self.obs[(i + self.nstep) % self.n], self.done[i]


# ----------------------------------------------------------------------------
# 2.5) 인간 데모 녹화 + Behavioral Cloning
# ----------------------------------------------------------------------------
class HumanRecorder:
    """사람이 스페이스를 누르는 동안 화면 스택 + 행동을 녹화한다."""
    def __init__(self, args):
        self.a = args
        self.stop = False
        self.space_held = False
        self.screen = Screen(args.crop)
        self.episodes = []          # list of (obs_list, act_list)
        self._load_existing()

    def _load_existing(self):
        path = self.a.demo
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            obs = d["obs"]          # list of arrays (N, CHANNELS, H, W)
            act = d["act"]
            for o, a in zip(obs, act):
                self.episodes.append((o, a))
            print(f"기존 데모 불러옴: {path}  ({len(self.episodes)} 에피소드, "
                  f"{sum(len(a) for _, a in self.episodes)} 프레임)")

    def _on_press(self, key):
        if key == keyboard.Key.space:
            self.space_held = True
        elif key == keyboard.Key.f8:
            self.stop = True

    def _on_release(self, key):
        if key == keyboard.Key.space:
            self.space_held = False

    def wait_for_start(self):
        t0, warned = time.time(), 0
        while not self.stop:
            _, p = self.screen.read()
            if p < self.a.start_below:
                self.screen.reset()
                return
            if time.time() - t0 > 8 and time.time() - warned > 8:
                print("  (진행률이 처음으로 안 돌아옴 - 레벨을 직접 다시 시작해 주세요)")
                warned = time.time()
            time.sleep(0.05)

    @staticmethod
    def sleep_until(t):
        while True:
            d = t - time.perf_counter()
            if d <= 0:
                return
            time.sleep(d - 0.001 if d > 0.002 else 0)

    def run_episode(self, first):
        a, dt = self.a, 1.0 / self.a.hz
        if not first:
            time.sleep(a.respawn_wait)
        self.wait_for_start()
        frames = collections.deque(maxlen=STACK)
        obs_l, act_l, prog_l = [], [], []
        game, prog = self.screen.read()
        for _ in range(STACK):
            frames.append(game)
        best, t_inc, k, outcome = prog, time.perf_counter(), 0, None
        t_next = t_start = time.perf_counter()
        while not self.stop:
            obs = make_obs(frames)
            act = 1 if self.space_held else 0
            obs_l.append(obs)
            act_l.append(act)
            prog_l.append(prog)
            t_next += dt
            self.sleep_until(t_next)
            if time.perf_counter() > t_next + 3 * dt:
                t_next = time.perf_counter()
            game, prog = self.screen.read()
            frames.append(game)
            now = time.perf_counter()
            if prog >= a.clear_at:
                outcome = "clear"
                break
            if prog > best + self.screen.min_step:
                best, t_inc, k = prog, now, len(obs_l)
            if now - t_inc > a.dead_window or prog < best - 0.05:
                outcome = "dead"
                break
        if outcome is None:
            return None
        n = len(obs_l)
        T = n if outcome == "clear" else min(k + 2, n)
        obs_arr = np.stack(obs_l[:T]).astype(np.uint8)  # (T, CHANNELS, H, W)
        act_arr = np.array(act_l[:T], dtype=np.int64)
        self.episodes.append((obs_arr, act_arr))
        return outcome, best, n / max(time.perf_counter() - t_start, 1e-6), T

    def save(self):
        if not self.episodes:
            print("저장할 데모가 없습니다.")
            return
        obs = np.array([e[0] for e in self.episodes], dtype=object)
        act = np.array([e[1] for e in self.episodes], dtype=object)
        np.savez_compressed(self.a.demo, obs=obs, act=act)
        total = sum(len(a) for _, a in self.episodes)
        jumps = sum(int(a.sum()) for _, a in self.episodes)
        print(f"데모 저장: {self.a.demo}  ({len(self.episodes)} 에피소드, {total} 프레임, "
              f"점프 비율 {jumps / max(total, 1) * 100:.1f}%)")

    def run(self):
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        print(f"{self.a.countdown}초 안에 게임 창을 클릭해서 포커스를 주세요.")
        print("사람이 평소처럼 플레이하면 됩니다. (스페이스=점프, F8=녹화 중지)")
        for i in range(self.a.countdown, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        ep = 0
        try:
            while not self.stop:
                res = self.run_episode(first=(ep == 0))
                if res is None:
                    break
                ep += 1
                outcome, best, hz, T = res
                print(f"데모 ep {ep:3d} | {'클리어!' if outcome == 'clear' else '사망  '} "
                      f"진행률 {best * 100:5.1f}% | {T} 프레임 | {hz:.0f}Hz", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop = True
            self.save()
            listener.stop()


def train_bc(args):
    """데모 데이터로 Behavioral Cloning (교차 엔트로피) 학습."""
    path = args.demo
    if not os.path.exists(path):
        sys.exit(f"데모 파일이 없습니다: {path}\n  먼저  python gd_real.py record  로 녹화하세요.")
    d = np.load(path, allow_pickle=True)
    obs_list, act_list = d["obs"], d["act"]
    all_obs = np.concatenate([o for o in obs_list], axis=0)
    all_act = np.concatenate([a for a in act_list], axis=0)
    if all_obs.ndim != 4 or all_obs.shape[1] != CHANNELS:
        sys.exit(
            f"데모 채널 수 불일치: 관측 shape={all_obs.shape}, 기대 채널={CHANNELS}.\n"
            f"  프레임 차분 추가 후에는  python gd_real.py record  로 데모를 다시 녹화하세요."
        )
    n = len(all_act)
    print(f"데모 로드: {n} 프레임, 점프 비율 {all_act.mean() * 100:.1f}%  (채널 {CHANNELS})")

    # 클래스 불균형 보정 (점프가 보통 적음)
    counts = np.bincount(all_act, minlength=2).astype(np.float64)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    print(f"클래스 가중치: 가만히={weights[0]:.2f}, 점프={weights[1]:.2f}")

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = QNet().to(dev)
    if args.load:
        ck = torch.load(args.load, map_location=dev)
        net.load_state_dict(ck["net"])
        print(f"초기 가중치 불러옴: {args.load}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    w = torch.tensor(weights, dtype=torch.float32, device=dev)

    epochs = args.bc_epochs
    batch = args.batch
    idx = np.arange(n)
    best_acc = 0.0
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        total_loss, correct, total = 0.0, 0, 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            o = torch.from_numpy(all_obs[b]).to(dev)
            a = torch.from_numpy(all_act[b]).to(dev)
            logits = net(o)
            loss = F.cross_entropy(logits, a, weight=w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            total_loss += float(loss.item()) * len(b)
            pred = logits.argmax(1)
            correct += int((pred == a).sum().item())
            total += len(b)
        acc = correct / max(total, 1)
        print(f"BC epoch {ep:3d}/{epochs}  loss {total_loss / n:.4f}  acc {acc * 100:.1f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({"net": net.state_dict(), "steps": 0, "bc_acc": acc}, args.save)
    print(f"BC 모델 저장: {args.save}  (최고 정확도 {best_acc * 100:.1f}%)")
    print("이제  python gd_real.py play --load", args.save, " 로 스스로 플레이해 보세요.")


# ----------------------------------------------------------------------------
# 3) 학습 루프
#    - 메인 스레드: 정해진 Hz로 화면 보고 -> 행동 -> 키 입력 (게임은 실시간이라 멈출 수 없음)
#    - 학습 스레드: 그 사이에 계속 배치 학습
# ----------------------------------------------------------------------------
class Trainer:
    def __init__(self, a):
        self.a, self.stop = a, False
        self.dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q, self.tgt, self.act_net = QNet().to(self.dev), QNet().to(self.dev), QNet().to(self.dev)
        self.env_steps = self.learn_steps = 0
        if a.load:
            ck = torch.load(a.load, map_location=self.dev)
            self.q.load_state_dict(ck["net"]); self.env_steps = ck.get("steps", 0)
            print(f"불러옴: {a.load} (누적 {self.env_steps} 스텝)")
        self.tgt.load_state_dict(self.q.state_dict()); self.act_net.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=a.lr)
        self.replay, self.net_lock = Replay(a.buffer, a.nstep), threading.Lock()
        self.screen, self.pad = Screen(a.crop), Pad()
        self.hist = collections.deque(maxlen=20)
        self.sticky_left, self.sticky_act, self.loss_ema = 0, 0, 0.0
        self.ep = 0

    # --- 행동 선택 ---
    def eps(self):
        if self.a.mode == "play":
            return 0.0
        return max(0.05, 1.0 - self.env_steps / self.a.eps_steps)

    def choose(self, stack):
        if self.sticky_left > 0:                                  # 진행 중인 탐험 행동 유지
            self.sticky_left -= 1
            return self.sticky_act
        if random.random() < self.eps():
            self.sticky_act = int(random.random() < self.a.jump_p)
            self.sticky_left = (random.randint(2, 8) if self.sticky_act else random.randint(3, 15)) - 1
            return self.sticky_act
        with torch.no_grad():
            x = torch.from_numpy(stack).unsqueeze(0).to(self.dev)
            return int(self.act_net(x).argmax(1).item())

    # --- 학습 스레드 ---
    def learner(self):
        a = self.a
        try:
            while not self.stop:
                if self.replay.size < a.min_replay or self.learn_steps >= self.env_steps * a.ratio:
                    time.sleep(0.003); continue
                o, ac, r, n, d = self.replay.sample(a.batch, a.near_frac)
                o, n = torch.from_numpy(o).to(self.dev), torch.from_numpy(n).to(self.dev)
                ac, r, d = [torch.from_numpy(v).to(self.dev) for v in (ac, r, d)]
                qsa = self.q(o).gather(1, ac[:, None]).squeeze(1)
                with torch.no_grad():                            # Double DQN
                    an = self.q(n).argmax(1, keepdim=True)
                    y = r + (a.gamma ** a.nstep) * (1 - d) * self.tgt(n).gather(1, an).squeeze(1)
                loss = F.smooth_l1_loss(qsa, y)
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
                with self.net_lock:
                    self.opt.step()
                with torch.no_grad():                            # 타깃 네트워크 부드럽게 갱신
                    for tp, qp in zip(self.tgt.parameters(), self.q.parameters()):
                        tp.mul_(1 - a.tau).add_(a.tau * qp)
                self.learn_steps += 1
                self.loss_ema = 0.99 * self.loss_ema + 0.01 * float(loss.item())
        except Exception as e:
            print("학습 스레드 오류:", repr(e)); self.stop = True

    def sync_act_net(self):
        with self.net_lock:
            self.act_net.load_state_dict(self.q.state_dict())

    def save(self):
        torch.save({"net": self.q.state_dict(), "steps": self.env_steps}, self.a.save)

    # --- 에피소드 진행 ---
    @staticmethod
    def sleep_until(t):
        while True:
            d = t - time.perf_counter()
            if d <= 0: return
            time.sleep(d - 0.001 if d > 0.002 else 0)

    def wait_for_start(self):
        """죽은 뒤 게임이 재시작되어 진행률이 0 근처로 돌아올 때까지 대기"""
        t0, warned = time.time(), 0
        while not self.stop:
            _, p = self.screen.read()
            if p < self.a.start_below:
                self.screen.reset()
                return
            if time.time() - t0 > 8 and time.time() - warned > 8:
                print("  (진행률이 처음으로 안 돌아옴 - 레벨 클리어 화면이면 레벨을 직접 다시 시작해 주세요)"); warned = time.time()
            time.sleep(0.05)

    def run_episode(self, first):
        a, dt = self.a, 1.0 / self.a.hz
        self.pad.set(0)
        self.sticky_left = 0
        if not first:
            time.sleep(a.respawn_wait)
        self.wait_for_start()
        obs_l, act_l, prog_l, frames = [], [], [], collections.deque(maxlen=STACK)
        game, prog = self.screen.read()
        for _ in range(STACK):
            frames.append(game)
        best, t_inc, k, outcome = prog, time.perf_counter(), 0, None
        t_next = t_start = time.perf_counter()
        while not self.stop:
            obs = make_obs(frames)
            act = self.choose(obs)
            self.pad.set(act)
            obs_l.append(obs); act_l.append(act); prog_l.append(prog)
            self.env_steps += 1
            if self.env_steps % 60 == 0:
                self.sync_act_net()
            t_next += dt
            self.sleep_until(t_next)
            if time.perf_counter() > t_next + 3 * dt:            # 너무 밀렸으면 기준 시각 재설정
                t_next = time.perf_counter()
            game, prog = self.screen.read(); frames.append(game)
            now = time.perf_counter()
            if prog >= a.clear_at:
                outcome = "clear"; break
            if prog > best + self.screen.min_step:
                best, t_inc, k = prog, now, len(obs_l)           # k: 진행률이 마지막으로 늘어난 시점의 관측 번호
            if now - t_inc > a.dead_window or prog < best - 0.05:
                outcome = "dead"; break                          # 진행률이 멈추거나 되돌아감 = 사망
        self.pad.set(0)
        if outcome is None:                                      # F8로 중단된 에피소드는 버림
            return None
        n = len(obs_l)
        T = n if outcome == "clear" else min(k + 2, n)          # 사망: 진행률이 멈춘 직후까지만 사용 (폭발 장면 제외)
        r1 = [(prog_l[i + 1] - prog_l[i]) * a.scale for i in range(T - 1)]
        r1.append(1.0 if outcome == "clear" else -a.death_penalty)   # 완주 +1 / 사망 -death_penalty
        N, g = a.nstep, a.gamma
        rew = [sum(g ** j * r1[i + j] for j in range(min(N, T - i))) for i in range(T)]   # n-step 누적 보상
        done = [0.0 if i + N < T else 1.0 for i in range(T)]                              # 종료까지 N 이내면 부트스트랩 X
        near = [outcome == "dead" and i >= T - a.near_steps for i in range(T)]
        if a.mode != "play":
            self.replay.add_episode(obs_l[:T], act_l[:T], rew, done, near)
        return outcome, best, n / (time.perf_counter() - t_start)

    def run(self):
        a = self.a
        listener = keyboard.Listener(on_press=lambda k: setattr(self, "stop", True) if k == keyboard.Key.f8 else None)
        listener.start()
        if a.mode == "train":
            threading.Thread(target=self.learner, daemon=True).start()
        print(f"{a.countdown}초 안에 게임 창을 클릭해서 포커스를 주세요. (중지: F8)")
        for i in range(a.countdown, 0, -1):
            print(f"  {i}...", flush=True); time.sleep(1)
        log = open("gd_real_log.csv", "a")
        try:
            while not self.stop and self.env_steps < a.steps:
                res = self.run_episode(first=(self.ep == 0))
                if res is None:
                    break
                self.ep += 1
                outcome, best, hz = res
                self.hist.append(best)
                print(f"ep {self.ep:5d} | {'클리어!' if outcome == 'clear' else '사망  '} 진행률 {best*100:5.1f}% | "
                      f"최근20 평균 {np.mean(self.hist)*100:5.1f}% | eps {self.eps():.2f} | "
                      f"steps {self.env_steps} | learn {self.learn_steps} | loss {self.loss_ema:.3f} | {hz:.0f}Hz", flush=True)
                if hz < 0.8 * a.hz:
                    print(f"  ⚠ 실제 루프 속도 {hz:.0f}Hz < 목표 {a.hz:.0f}Hz : 캡처가 느림. 게임 창을 작게 하거나 --hz 를 낮추세요.")
                log.write(f"{self.ep},{self.env_steps},{best:.4f},{outcome}\n"); log.flush()
                if a.mode == "train" and self.ep % 10 == 0:
                    self.save()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop = True
            self.pad.set(0)
            if a.mode == "train":
                self.save(); print(f"모델 저장: {a.save}")
            listener.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["calibrate", "test", "train", "play", "record", "bc"])
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--hz", type=float, default=30, help="초당 행동 횟수")
    p.add_argument("--steps", type=int, default=2_000_000, help="최대 환경 스텝")
    p.add_argument("--load", default=None)
    p.add_argument("--save", default=None, help="기본값: train→gd_real.pt, bc→gd_bc.pt")
    p.add_argument("--demo", default=DEMO_PATH, help="데모 저장/로드 경로")
    p.add_argument("--device", default=None)
    p.add_argument("--dead_window", type=float, default=0.6, help="진행률이 이 시간(초) 동안 안 늘면 사망으로 판단")
    p.add_argument("--respawn_wait", type=float, default=1.2, help="사망 후 재시작 대기(초)")
    p.add_argument("--start_below", type=float, default=0.05)
    p.add_argument("--clear_at", type=float, default=0.99)
    p.add_argument("--scale", type=float, default=10.0, help="진행률 1.0당 보상")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=30000, help="리플레이 크기 (메모리 약 0.5GB)")
    p.add_argument("--min_replay", type=int, default=3000)
    p.add_argument("--ratio", type=float, default=1.0, help="환경 스텝 1번당 학습 스텝 수")
    p.add_argument("--eps_steps", type=int, default=40000, help="탐험률이 1.0 -> 0.05로 줄어드는 스텝 수")
    p.add_argument("--jump_p", type=float, default=0.3, help="무작위 탐험 때 점프를 선택할 확률")
    p.add_argument("--nstep", type=int, default=5, help="n-step 리턴 (죽음/진행 결과를 더 빨리 전파)")
    p.add_argument("--death_penalty", type=float, default=3.0)
    p.add_argument("--near_frac", type=float, default=0.3, help="배치 중 '죽기 직전' 장면 비율")
    p.add_argument("--near_steps", type=int, default=60, help="'죽기 직전'으로 보는 스텝 수 (30Hz면 2초)")
    p.add_argument("--crop", type=float, nargs=4, default=[0.1, 0.65, 0.0, 1.0], metavar=("X0", "X1", "Y0", "Y1"),
                   help="게임 화면에서 사용할 비율 영역 (플레이어+앞쪽만)")
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--bc_epochs", type=int, default=30, help="BC 학습 에폭 수")
    args = p.parse_args()

    if args.save is None:
        args.save = "gd_bc.pt" if args.mode == "bc" else "gd_real.pt"

    if args.mode == "calibrate":
        calibrate(args.monitor)
    elif args.mode == "test":
        test(args)
    elif args.mode == "record":
        HumanRecorder(args).run()
    elif args.mode == "bc":
        train_bc(args)
    else:
        if args.mode == "play" and not args.load:
            sys.exit("play 모드는 --load gd_real.pt (또는 gd_bc.pt) 가 필요합니다.")
        Trainer(args).run()