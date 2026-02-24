# Mac WebCam Pose Estimation (MediaPipe x YOLOv8)

Mac（特にApple Silicon M1/M2/M3）のWebカメラ映像から、複数人の骨格をリアルタイムで高精度に推定するPythonプロトタイプです。
YOLOv8による高速な人物検出と、MediaPipe Poseによる骨格推定を組み合わせています。

## 特徴 (Features)
- **複数人対応**: YOLOv8を用いて画面内の人物を全て検出し、個別にMediaPipeを適用することで複数人の同時骨格推定を実現しています。
- **Mac環境への最適化 (CPU動作)**: NVIDIA GPUのないApple Silicon搭載Mac環境でも、軽量化されたモデルを使用し高速に動作します。（Mac環境特有の依存関係エラーを回避するため、安定版のMediaPipe v0.10.14を採用しています）。
- **リアルタイム処理**: Webカメラ映像を読み込みながら、遅延なく（数十FPS）骨格を描画します。

## 動作環境 (Requirements)
- macOS (Apple Silicon M1/M2/M3推奨)
- Python 3.9 以上
- Webカメラ（内蔵カメラでOK）

## インストール方法 (Installation)

1. リポジトリのクローン
```bash
git clone https://github.com/kyopan/mediapipe_prototype.git
cd mediapipe_prototype
```

2. 仮想環境の作成と有効化
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```

## 使い方 (Usage)

#### 1. 単体（1人）用プロトタイプの実行
自撮り用などで1人だけを推定する場合はこちらを実行します。
```bash
python main.py
```

#### 2. 複数人対応プロトタイプの実行
複数人を同時に検出・推定する場合はこちらを実行します。
```bash
python multi_person_pose.py
```

※ 初回実行時は、YOLOv8の軽量学習済みモデル `yolov8n.pt` が自動的にダウンロードされます（約6MB）。
※ macOSの初回起動時、「ターミナルからカメラへのアクセス許可」を求めるポップアップが表示された場合は「OK」をクリックしてください。

## 終了方法
カメラ映像が映っているウィンドウを選択した状態で、キーボードの `q` キーを押すと終了します。
