import cv2
import mediapipe as mp
from ultralytics import YOLO

def main():
    # 超高速な物体検出AI「YOLOv8 Nano」モデルを読み込む
    # （初回実行時のみ、軽量な学習済みモデルデータが自動でダウンロードされます）
    model = YOLO('yolov8n.pt') 

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # MediaPipe Poseの初期化
    # 複数人の切り抜き画像を次々処理するため、static_image_mode=True が安定します
    pose = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=1,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: Webカメラを開けませんでした。")
        return

    print("=========================================")
    print("YOLO x MediaPipe 複数人検知を開始します！")
    print("（映像ウィンドウをアクティブにして 'q' キーで終了）")
    print("=========================================")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 先に YOLOv8 で画像内の人物を探す (classes=0 は「人=person」のみを指定)
        # verbose=False でターミナルのログ出力をスッキリさせます
        results = model.predict(image, classes=0, verbose=False)

        # 骨格と枠を描画するためのキャンバスを用意
        annotated_image = image.copy()

        # 検出されたすべての人の情報をループ処理
        for r in results:
            for box in r.boxes:
                # 1人分の枠（バウンディングボックス）の座標を取得
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 枠が画面外にはみ出ないようにクリップ（エラー防止）
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)

                # 枠が小さすぎる（誤検出等の）場合はスキップ
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue 
                    
                # 1人の領域だけを切り出す (ROI: Region of Interest)
                person_roi = image[y1:y2, x1:x2]
                person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

                # MediaPipeで「その1人」だけの骨格を推定する
                pose_results = pose.process(person_roi_rgb)

                # 骨格が見つかった場合
                if pose_results.pose_landmarks:
                    # 描画対象を全体の画像の一部（切り出した範囲）に向ける
                    drawn_roi = annotated_image[y1:y2, x1:x2]
                    
                    # その人の範囲にだけ骨格線を描画
                    mp_drawing.draw_landmarks(
                        drawn_roi,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                # （オプション）YOLOが検知した「人」の枠を緑色で四角く囲む
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 左右反転（自撮り鏡面）にして表示
        cv2.imshow('YOLO x MediaPipe - Multi Person', cv2.flip(annotated_image, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソース解放
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
