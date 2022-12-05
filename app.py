#!/usr/bin/env python

import csv
import copy
import argparse
import itertools
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
        '-im',
        '--image',
        type=str,
        default='',
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

    if not args.image:
        cap_device = args.device
    else:
        cap_device = args.image
    cap_width = args.width
    cap_height = args.height
    min_detection_confidence = args.min_detection_confidence

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

    # 座標履歴 ################################################################
    history_length = 16
    # point_history = deque(maxlen=history_length)
    point_history = {}
    pre_point_history = {}

    # フィンガージェスチャー履歴 #################################################
    gesture_history_length = 10
    finger_gesture_history = {}

    # 手のひらトラッキング用手のひら中心座標最新履歴 #################################
    # {
    #   int(trackid1): [cx, cy],
    #   int(trackid2): [cx, cy],
    #   int(trackid3): [cx, cy],
    #     :
    # }
    palm_trackid_cxcy = {}

    #  #######################################################################
    mode = 0
    wh_ratio = cap_width / cap_height

    auto = False
    prev_number = -1
    image = None

    while True:
        fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1) if not args.image else cv.waitKey(0) if image is not None and args.image else cv.waitKey(1)
        if key == 27:  # ESC
            break
        number, mode, auto, prev_number = select_mode(key, mode, auto, prev_number)

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

        # 手の検出件数がゼロになったらトラッキング用手のひら中心座標最新履歴を初期化
        if len(hands) == 0:
            palm_trackid_cxcy = {}
        # トラッキング用手のひら中心座標最新履歴とバウンディングボックスの検出順序紐づけリスト
        palm_trackid_box_x1y1s = {}

        if len(hands) > 0:
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

            # 回転角度をゼロ度に補正した手のひら画像の取得
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
                cv.putText(
                    debug_image,
                    f'{y2-y1}x{x2-x1}',
                    (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,0),
                    2,
                    cv.LINE_AA,
                )
                cv.putText(
                    debug_image,
                    f'{y2-y1}x{x2-x1}',
                    (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (59,255,255),
                    1,
                    cv.LINE_AA,
                )
                # 検出枠の描画
                cv.rectangle(
                    debug_image,
                    (x1,y1),
                    (x2,y2),
                    (0,128,255),
                    2,
                    cv.LINE_AA,
                )
                # 検出領域の中心座標描画
                cv.circle(
                    debug_image,
                    (rcx, rcy),
                    3,
                    (0, 255, 255),
                    -1,
                )
                """
                手のひらトラッキング用手のひら中心座標最新履歴の保存
                    1. 過去履歴の中から基準点との距離が一番近い中心座標を抽出
                    2. 距離が100pxより離れている場合は新たな手のひらと認識させる
                    3. 距離が100px以下の場合は該当のtrackidを割り当てて過去履歴の中心座標を上書きする
                """
                # 1. 過去履歴の中から基準点との距離が一番近い中心座標を抽出
                base_point = np.asarray(
                    [rcx, rcy],
                    dtype=np.float32,
                )
                points = np.asarray(
                    list(palm_trackid_cxcy.values()),
                    dtype=np.float32,
                )
                if len(points) > 0:
                    # 最近傍点探索
                    diff_val = points - base_point
                    all_points_distance = np.linalg.norm(diff_val, axis=1)
                    nearest_trackid = np.argmin(all_points_distance)
                    nearest_distance = all_points_distance[nearest_trackid]
                    new_trackid = int(nearest_trackid) + 1
                    # 2. 距離が100pxより離れている場合は新たな手のひらと認識させる
                    # 3. 距離が100px以下の場合は該当のtrackidを割り当てて過去履歴の中心座標を上書きする
                    if nearest_distance > 100:
                        # 現状のtrackid最大値+1を新規trackidとして生成
                        new_trackid = next(iter(reversed(palm_trackid_cxcy))) + 1
                else:
                    # trackid初期値
                    new_trackid = 1

                # 手のひらトラッキング用手のひら中心座標最新履歴の最新座標を更新 または 新規追加
                palm_trackid_cxcy[new_trackid] = [rcx, rcy]
                # バウンディングボックスの検出順序とtrackidの順序を整合
                # box_x1y1x2y2_palm_trackids.append([x1, y1, x2, y2, new_trackid])
                palm_trackid_box_x1y1s[new_trackid] = [x1, y1]
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
                pre_processed_landmarks = []
                pre_processed_point_histories = []
                for (trackid, x1y1), landmark, rotated_image_size_leftright, not_rotate_rect in \
                    zip(palm_trackid_box_x1y1s.items(), hand_landmarks, rotated_image_size_leftrights, not_rotate_rects):

                    x1, y1 = x1y1
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
                    text_y = max(y1-70, 20)
                    text_y = min(text_y, cap_height-70)
                    cv.putText(
                        debug_image,
                        f'trackid:{trackid} {handedness}',
                        (text_x, text_y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,0),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        debug_image,
                        f'trackid:{trackid} {handedness}',
                        (text_x, text_y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (59,255,255),
                        1,
                        cv.LINE_AA,
                    )

                    # 相対座標・正規化座標への変換
                    """
                    pre_processed_landmark: np.ndarray [42], [x,y]x21
                    """
                    pre_processed_landmark = pre_process_landmark(
                        landmark,
                    )
                    pre_processed_landmarks.append(pre_processed_landmark)

                """
                point_history: dict
                {
                    int(trackid1): [[x, y],[x, y],[x, y],[x, y], ...],
                    int(trackid2): [[x, y],[x, y], ...],
                    int(trackid3): [[x, y],[x, y],[x, y], ...],
                        :
                }
                    ↓
                pre_processed_point_histories: List
                [
                    [rx, ry, rx, ry, rx, ry, rx, ry, ...],
                    [rx, ry, rx, ry, ...],
                    [rx, ry, rx, ry, rx, ry, ...],
                        :
                ]
                """
                # 人差し指軌跡を相対座標へ変換
                pre_processed_point_histories = pre_process_point_history(
                    image_width=debug_image.shape[1],
                    image_height=debug_image.shape[0],
                    point_history=point_history,
                )

                # 学習データ保存
                logging_csv(
                    number,
                    mode,
                    trackid,
                    pre_processed_landmark,
                    pre_processed_point_histories,
                )

                # ハンドサイン分類 - バッチ処理
                hand_sign_ids = keypoint_classifier(
                    np.asarray(pre_processed_landmarks, dtype=np.float32)
                )
                for (trackid, x1y1), landmark, hand_sign_id in zip(palm_trackid_box_x1y1s.items(), hand_landmarks, hand_sign_ids):
                    x1, y1 = x1y1
                    point_history.setdefault(trackid, deque(maxlen=history_length))
                    if hand_sign_id == 2:  # 指差しサイン
                        point_history[trackid].append(list(landmark[8])) # 人差指座標
                    else:
                        point_history[trackid].append([0, 0])

                """
                人差し指の軌跡が表示上に残り続けるのを割けるため
                トラッキング対象外になった(画角から外れた)手のひらがある場合は人差指XY座標の履歴をクリアする
                今回の全ての軌跡座標と前回の全ての軌跡座標が完全に一致したtrackidの履歴情報は変化なしと断定して履歴から削除する
                point_history: 最新の軌跡16点
                pre_point_history: 前回の軌跡16点
                """
                if len(pre_point_history) > 0:
                    temp_point_history = copy.deepcopy(point_history)
                    for track_id, points in temp_point_history.items():
                        if track_id in pre_point_history:
                            pre_points = pre_point_history[track_id]
                            if points == pre_points:
                                _ = point_history.pop(track_id, None)
                pre_point_history = copy.deepcopy(point_history)

                # フィンガージェスチャー分類 - バッチ処理
                finger_gesture_ids = None
                temp_trackid_x1y1s = {}
                temp_pre_processed_point_history = []
                for (trackid, x1y1), pre_processed_point_history in zip(palm_trackid_box_x1y1s.items(), pre_processed_point_histories):
                    point_history_len = len(pre_processed_point_history)
                    if point_history_len > 0 and point_history_len % (history_length * 2) == 0:
                        temp_trackid_x1y1s[trackid] = x1y1
                        temp_pre_processed_point_history.append(pre_processed_point_history)
                if len(temp_pre_processed_point_history) > 0:
                    finger_gesture_ids = point_history_classifier(
                        temp_pre_processed_point_history,
                    )

                # 直近検出の中で最多のジェスチャーIDを算出
                if finger_gesture_ids is not None:
                    for (trackid, x1y1), finger_gesture_id in zip(temp_trackid_x1y1s.items(), finger_gesture_ids):
                        x1, y1 = x1y1
                        trackid_str = str(trackid)
                        finger_gesture_history.setdefault(trackid_str, deque(maxlen=gesture_history_length))
                        finger_gesture_history[trackid_str].append(int(finger_gesture_id))
                        most_common_fg_id = Counter(finger_gesture_history[trackid_str]).most_common()
                        text_x = max(x1, 10)
                        text_x = min(text_x, cap_width-120)
                        text_y = max(y1-45, 20)
                        text_y = min(text_y, cap_height-45)
                        classifier_label = point_history_classifier_labels[most_common_fg_id[0][0]]
                        # print(f'trackid: {trackid} [x1,y1]: [{x1},{y1}] finger_gesture_id: {classifier_label}')
                        cv.putText(
                            debug_image,
                            f'{classifier_label}',
                            (text_x, text_y),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,0),
                            2,
                            cv.LINE_AA,
                        )
                        cv.putText(
                            debug_image,
                            f'{classifier_label}',
                            (text_x, text_y),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (59,255,255),
                            1,
                            cv.LINE_AA,
                        )

            else:
                point_history = {}

        else:
            point_history = {}

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number, auto)

        # 画面反映 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        video_writer.write(debug_image)

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode, auto=False, prev_number=-1):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
        prev_number = number
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 97:   # a
        auto = not auto
    if auto == True:
        if prev_number != -1:
            number = prev_number
    else:
        prev_number = -1

    return number, mode, auto, prev_number


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


def pre_process_point_history(
    image_width: int,
    image_height: int,
    point_history: dict,
):
    """pre_process_point_history

    Parameters
    ----------
    image_width: int
        Input image width

    image_height: int
        Input image height

    point_history: dict
        Index finger XY coordinate history per trackid (detected palm)
        {
            int(trackid1): [[x, y],[x, y],[x, y],[x, y], ...],
            int(trackid2): [[x, y],[x, y], ...],
            int(trackid3): [[x, y],[x, y],[x, y], ...],
                :
        }

    Returns
    -------
    relative_coordinate_list_by_trackid: List
        [
            [rx, ry, rx, ry, rx, ry, rx, ry, ...],
            [rx, ry, rx, ry, ...],
            [rx, ry, rx, ry, rx, ry, ...],
                :
        ]
    """
    if len(point_history) == 0:
        return []

    temp_point_history = copy.deepcopy(point_history)
    relative_coordinate_list_by_trackid = []

    # trackidごとに相対座標へ変換
    for trackid, points in temp_point_history.items():
        base_x, base_y = points[0][0], points[0][1]
        relative_coordinate_list = [
            [
                (point[0] - base_x) / image_width,
                (point[1] - base_y) / image_height,
            ] for point in points
        ]
        # 1次元リストに変換
        relative_coordinate_list_1d = list(
            itertools.chain.from_iterable(relative_coordinate_list)
        )
        relative_coordinate_list_by_trackid.append(relative_coordinate_list_1d)
    return relative_coordinate_list_by_trackid


def logging_csv(number, mode, trackid, landmark_list, point_histories):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, trackid, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            for point_history in point_histories:
                writer.writerow([number, trackid, *point_history])


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
    _ = [
        cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2) \
            for trackid, points in point_history.items() \
                for index, point in enumerate(points) if point[0] != 0 and point[1] != 0
    ]
    return image


def draw_info(image, fps, mode, number, auto):
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
    cv.putText(
        image,
        f'AUTO:{str(auto)}',
        (10, 130),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return image


if __name__ == '__main__':
    main()
