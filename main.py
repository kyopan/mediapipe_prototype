import cv2
import mediapipe as mp

def main():
    # MediaPipeのPose（姿勢推定）モジュールを初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Macの内蔵Webカメラ（0番）を起動
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("エラー: Webカメラを開けませんでした。")
        return

    print("=========================================")
    print("カメラを起動しました。（映像ウィンドウをフォーカスして 'q' キーで終了）")
    print("=========================================")

    # Poseモデルの設定
    with mp_pose.Pose(
        min_detection_confidence=0.5, # 検出の信頼度の閾値（0.0 ~ 1.0）
        min_tracking_confidence=0.5   # 追跡の信頼度の閾値（0.0 ~ 1.0）
    ) as pose:
        
        while cap.isOpened():
            # カメラからフレーム（1コマの画像）を読み込む
            success, image = cap.read()
            if not success:
                print("空のカメラフレームを無視しました。")
                continue

            # パフォーマンス向上のため、画像を読み取り専用にする
            image.flags.writeable = False
            
            # OpenCVはBGR形式で画像を読み込むため、MediaPipe用のRGB形式に変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipeに画像を渡して姿勢推定を実行！
            results = pose.process(image)

            # 画像に骨格線を描画するために、再度書き込み可能に戻し、BGR形式に戻す
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 検出結果（ランドマーク）があれば、画像上に骨格を描画する
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,             # 検出された各関節の座標
                    mp_pose.POSE_CONNECTIONS,           # 関節同士を結ぶ線
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style() # デフォルトのスタイル
                )

            # 自撮り感覚で操作しやすいように、画像を左右反転（鏡面）にして表示
            cv2.imshow('MediaPipe Pose - Mac Web Camera', cv2.flip(image, 1))

            # 5ミリ秒待機し、'q' キーが押されたらループを終了する
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # カメラのリソースを解放し、すべてのウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
