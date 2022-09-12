#!/usr/bin/env python

import csv
import copy
import argparse
import itertools
from typing import List
from math import degrees
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from utils.utils import rotate_and_crop_rectangle
from model import PalmDetection
from model import HandLandmark
from model import KeyPointClassifier
from model import PointHistoryClassifier



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-wi',
        '--width',
        help='cap width',
        type=int,
        default=640,
    )
    parser.add_argument(
        '-he',
        '--height',
        help='cap height',
        type=int,
        default=480,
    )
    parser.add_argument(
        '-mdc',
        '--min_detection_confidence',
        help='min_detection_confidence',
        type=float,
        default=0.6,
    )
    parser.add_argument(
        '-mtc',
        '--min_tracking_confidence',
        help='min_tracking_confidence',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '-dif',
        '--disable_image_flip',
        help='disable image flip',
        action='store_true',
    )


    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    lines_hand = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],[0,17],
    ]

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # モデルロード #############################################################
    palm_detection = PalmDetection(score_threshold=min_detection_confidence)
    hand_landmark = HandLandmark()

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # ラベル読み込み ###########################################################
    with open(
        'model/keypoint_classifier/keypoint_classifier_label.csv',
        encoding='utf-8-sig',
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig',
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 座標履歴 #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # フィンガージェスチャー履歴 ################################################
    # finger_gesture_history = {deque(maxlen=history_length)}
    finger_gesture_history = {}
    finger_gesture_history.setdefault('0', deque(maxlen=history_length))

    #  ########################################################################
    mode = 0
    wh_ratio = cap_width / cap_height

    while True:
        fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = image if args.disable_image_flip else cv.flip(image, 1) # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################

        # ============================================================= PalmDetection
        # ハンドディテクション - シングルバッチ処理
        hands = palm_detection(image)
        # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y

        rects = []
        not_rotate_rects = []
        rects_tuple = None
        cropted_rotated_hands_images = []

        if len(hands) > 0:
            # Draw
            for hand in hands:
                # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
                sqn_rr_size = hand[0]
                rotation = hand[1]
                sqn_rr_center_x = hand[2]
                sqn_rr_center_y = hand[3]

                cx = int(sqn_rr_center_x * cap_width)
                cy = int(sqn_rr_center_y * cap_height)
                xmin = int((sqn_rr_center_x - (sqn_rr_size / 2)) * cap_width)
                xmax = int((sqn_rr_center_x + (sqn_rr_size / 2)) * cap_width)
                ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2)) * cap_height)
                ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2)) * cap_height)
                xmin = max(0, xmin)
                xmax = min(cap_width, xmax)
                ymin = max(0, ymin)
                ymax = min(cap_height, ymax)
                degree = degrees(rotation)
                # [boxcount, cx, cy, width, height, degree]
                rects.append([cx, cy, (xmax-xmin), (ymax-ymin), degree])

            rects = np.asarray(rects, dtype=np.float32)

            cropted_rotated_hands_images = rotate_and_crop_rectangle(
                image=image,
                rects_tmp=rects,
                operation_when_cropping_out_of_range='padding',
            )

            # Debug ===============================================================
            for rect in rects:
                # 回転考慮の領域の描画, 赤色の枠
                rects_tuple = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
                box = cv.boxPoints(rects_tuple).astype(np.int0)
                cv.drawContours(debug_image, [box], 0,(0,0,255), 2, cv.LINE_AA)

                # 回転非考慮の領域の描画, オレンジ色の枠
                rcx = int(rect[0])
                rcy = int(rect[1])
                half_w = int(rect[2] // 2)
                half_h = int(rect[3] // 2)
                x1 = rcx - half_w
                y1 = rcy - half_h
                x2 = rcx + half_w
                y2 = rcy + half_h
                text_x = max(x1, 10)
                text_x = min(text_x, cap_width-120)
                text_y = max(y1-15, 45)
                text_y = min(text_y, cap_height-20)
                # [boxcount, rcx, rcy, x1, y1, x2, y2, height, degree]
                not_rotate_rects.append([rcx, rcy, x1, y1, x2, y2, 0])
                # 検出枠のサイズ WxH
                cv.putText(debug_image, f'{y2-y1}x{x2-x1}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv.LINE_AA)
                cv.putText(debug_image, f'{y2-y1}x{x2-x1}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (59,255,255), 1, cv.LINE_AA)
                # 検出枠の描画
                cv.rectangle(debug_image, (x1,y1), (x2,y2), (0,128,255), 2, cv.LINE_AA)
                # 検出領域の中心座標描画
                cv.circle(debug_image, (rcx, rcy), 3, (0, 255, 255), -1)
                # Debug ===============================================================

        # ============================================================= HandLandmark
        if len(cropted_rotated_hands_images) > 0:

            # Inference HandLandmark - バッチ処理
            hand_landmarks, rotated_image_size_leftrights = hand_landmark(
                images=cropted_rotated_hands_images,
                rects=rects,
            )

            if len(hand_landmarks) > 0:
                # Draw
                pre_processed_landmark_list = []
                pre_processed_point_history_list = []
                for hand_idx, (landmark, rotated_image_size_leftright, not_rotate_rect) in enumerate(zip(hand_landmarks, rotated_image_size_leftrights, not_rotate_rects)):

                    rotated_image_width, _, left_hand_0_or_right_hand_1 = rotated_image_size_leftright
                    thick_coef = rotated_image_width / 400
                    lines = np.asarray(
                        [
                            np.array([landmark[point] for point in line]).astype(np.int32) for line in lines_hand
                        ]
                    )
                    radius = int(1+thick_coef*5)
                    cv.polylines(
                        debug_image,
                        lines,
                        False,
                        (255, 0, 0),
                        int(radius),
                        cv.LINE_AA,
                    )
                    _ = [cv.circle(debug_image, (int(x), int(y)), radius, (0,128,255), -1) for x,y in landmark[:,:2]]
                    left_hand_0_or_right_hand_1 = left_hand_0_or_right_hand_1 if args.disable_image_flip else (1 - left_hand_0_or_right_hand_1)
                    handedness = 'Left ' if left_hand_0_or_right_hand_1 == 0 else 'Right'
                    _, _, x1, y1, _, _, _ = not_rotate_rect
                    text_x = max(x1, 10)
                    text_x = min(text_x, cap_width-120)
                    text_y = max(y1-40, 20)
                    text_y = min(text_y, cap_height-40)
                    cv.putText(debug_image, f'{handedness}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv.LINE_AA)
                    cv.putText(debug_image, f'{handedness}', (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (59,255,255), 1, cv.LINE_AA)

                    # 相対座標・正規化座標への変換
                    """
                    pre_processed_landmark: np.ndarray [42], [x,y]x21
                    """
                    pre_processed_landmark = pre_process_landmark(
                        landmark,
                    )
                    pre_processed_landmark_list.append(pre_processed_landmark)
                    pre_processed_point_history = pre_process_point_history(
                        debug_image.shape[1],
                        debug_image.shape[0],
                        point_history,
                    )
                    pre_processed_point_history_list.append(pre_processed_point_history)

                    # 学習データ保存
                    logging_csv(
                        number,
                        mode,
                        hand_idx,
                        pre_processed_landmark,
                        pre_processed_point_history,
                    )

                # print(f'len(pre_processed_point_history_list): {len(pre_processed_point_history_list)}')

                # ハンドサイン分類 - バッチ処理
                hand_sign_ids = keypoint_classifier(
                    np.asarray(pre_processed_landmark_list, dtype=np.float32)
                )
                for hand_idx, (landmark, hand_sign_id) in enumerate(zip(hand_landmarks, hand_sign_ids)):
                    if hand_sign_id == 2:  # 指差しサイン
                        point_history.append(landmark[8]) # 人差指座標
                        # print(f'hand_idx: {hand_idx} keypoint_classifier_labels: {keypoint_classifier_labels[hand_sign_id]}')
                    else:
                        point_history.append([0, 0])


                """2点間の距離算出
                https://www.higashisalary.com/entry/numpy-linalg-norm
                import numpy as np
                a=np.array([1,2])
                b=np.array([2,3])
                distance=np.linalg.norm(b-a)
                print(distance)
                """

                """複数点間の距離一括算出
                https://teratail.com/questions/153138
                """


                # フィンガージェスチャー分類 - バッチ処理
                finger_gesture_ids = []
                pre_processed_point_history_list = np.asarray(pre_processed_point_history_list, dtype=np.float32)
                """
                hands = 2
                pre_processed_point_history_list.shape: (2, 32)
                """
                print(f'pre_processed_point_history_list.shape: {pre_processed_point_history_list.shape}')
                point_history_len = pre_processed_point_history_list.size
                # print(f'point_history_len: {point_history_len}')
                if point_history_len % (history_length * 2) == 0:
                    finger_gesture_ids = point_history_classifier(
                        pre_processed_point_history_list,
                    )
                    # print(f'finger_gesture_ids.shape: {finger_gesture_ids.shape}')

                # 直近検出の中で最多のジェスチャーIDを算出
                for hand_idx, finger_gesture_id in enumerate(finger_gesture_ids):
                    hand_idx_str = str(hand_idx)
                    finger_gesture_history.setdefault(str(hand_idx), deque(maxlen=history_length))
                    finger_gesture_history[hand_idx_str].append(int(finger_gesture_id))
                    most_common_fg_id = Counter(finger_gesture_history[hand_idx_str]).most_common()
                    print(f'hand_idx: {hand_idx} point_history_classifier_labels: {point_history_classifier_labels[most_common_fg_id[0][0]]}')

                    # # 描画
                    # debug_image = draw_info_text(
                    #     debug_image,
                    #     brect,
                    #     handedness,
                    #     keypoint_classifier_labels[hand_sign_id],
                    #     point_history_classifier_labels[most_common_fg_id[0][0]],
                    # )
            else:
                point_history.append([0, 0])

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # 画面反映 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        video_writer.write(debug_image)

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def pre_process_landmark(landmark_list):
    if len(landmark_list) == 0:
        return []

    temp_landmark_list = copy.deepcopy(landmark_list)
    # 相対座標に変換
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list = [
        [temp_landmark[0] - base_x, temp_landmark[1] - base_y] for temp_landmark in temp_landmark_list
    ]
    # 1次元リストに変換
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list)
    )
    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image_width, image_height, point_history):
    if len(point_history) == 0:
        return []

    temp_point_history = copy.deepcopy(point_history)
    # 相対座標に変換
    base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]
    temp_point_history = [
        [
            (point[0] - base_x) / image_width,
            (point[1] - base_y) / image_height,
        ] for point in temp_point_history
    ]
    # 1次元リストに変換
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history)
    )
    return temp_point_history


def logging_csv(number, mode, hand_idx, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, hand_idx, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, hand_idx, *point_history_list])
    return


def draw_info_text(
    image,
    brect,
    handedness,
    hand_sign_text,
    finger_gesture_text
):
    info_text = handedness
    if hand_sign_text != "":
        info_text = f'{handedness}:{hand_sign_text}'
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            f'Finger Gesture:{finger_gesture_text}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            f'Finger Gesture:{finger_gesture_text}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image,
                (point[0], point[1]),
                1 + int(index / 2),
                (152, 251, 152),
                2,
            )

    return image


def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(
            image,
            f'MODE:{mode_string[mode - 1]}',
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                f'NUM:{str(number)}',
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image


if __name__ == '__main__':
    main()
