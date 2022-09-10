import cv2
import math
import copy
import numpy as np
from typing import Tuple, List
from math import sin, cos, pi, floor


def normalize_radians(
    angle: float
) -> float:
    """__normalize_radians

    Parameters
    ----------
    angle: float

    Returns
    -------
    normalized_angle: float
    """
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def is_inside_rect(
    rects: np.ndarray,
    width_of_outer_rect: int,
    height_of_outer_rect: int,
) -> np.ndarray:
    """Determines whether rects is inside or outside the outer rectangle.

    Parameters
    ----------
    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Area to be verified.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    width_of_outer_rect: int
        Width of outer rectangle.

    height_of_outer_rect: int
        Height of outer rectangle

    Returns
    -------
    result: np.ndarray
        True: if the rotated sub rectangle is side the up-right rectange, False: else
    """
    results = []

    for rect in rects:
        cx = rect[0]
        cy = rect[1]
        width = rect[2]
        height = rect[3]
        angle = rect[4]

        if (cx < 0) or (cx > width_of_outer_rect):
            # Center X coordinate is outside the range of the outer rectangle
            results.append(False)

        elif (cy < 0) or (cy > height_of_outer_rect):
            # Center Y coordinate is outside the range of the outer rectangle
            results.append(False)

        else:
            # Coordinate acquisition of bounding rectangle considering rotation
            # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b
            rect_tuple = ((cx, cy), (width, height), angle)
            box = cv2.boxPoints(rect_tuple)

            x_max = int(np.max(box[:,0]))
            x_min = int(np.min(box[:,0]))
            y_max = int(np.max(box[:,1]))
            y_min = int(np.min(box[:,1]))

            if (x_min >= 0) and (x_max <= width_of_outer_rect) and \
                (y_min >= 0) and (y_max <= height_of_outer_rect):
                # All 4 vertices are within the perimeter rectangle
                results.append(True)
            else:
                # Any of the 4 vertices is outside the perimeter rectangle
                results.append(False)

    return np.asarray(results, dtype=np.bool_)


def bounding_box_from_rotated_rect(
    rects: np.ndarray,
) -> np.ndarray:
    """Conversion to bounding rectangle without rotation.

    Parameters
    ----------
    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Rotated rectangle.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    Returns
    -------
    result: np.ndarray
        e.g.:\n
        [input] rotated rectangle:\n
            [center:(10, 10), height:4, width:4, angle:45 degree]\n
        [output] bounding box for this rotated rectangle:\n
            [center:(10, 10), height:4*sqrt(2), width:4*sqrt(2), angle:0 degree]
    """
    results = []

    for rect in rects:
        cx = rect[0]
        cy = rect[1]
        width = rect[2]
        height = rect[3]
        angle = rect[4]

        rect_tuple = ((cx, cy), (width, height), angle)
        box = cv2.boxPoints(rect_tuple)

        x_max = int(np.max(box[:,0]))
        x_min = int(np.min(box[:,0]))
        y_max = int(np.max(box[:,1]))
        y_min = int(np.min(box[:,1]))

        cx = int((x_min + x_max) // 2)
        cy = int((y_min + y_max) // 2)
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        angle = 0
        results.append([cx, cy, width, height, angle])

    return np.asarray(results, dtype=np.float32)


def image_rotation_without_crop(
    images: List[np.ndarray],
    angles: np.ndarray,
) -> List[np.ndarray]:
    """Conversion to bounding rectangle without rotation.

    Parameters
    ----------
    images: List[np.ndarray]
        Image to be rotated.

    angles: np.ndarray
        Rotation degree.

    Returns
    -------
    rotated_images: List[np.ndarray]
        Image after rotation.
    """
    rotated_images = []
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    for image, angle in zip(images, angles):
        height, width = image.shape[:2]
        image_center = (width//2, height//2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, int(angle), 1)
        abs_cos = abs(rotation_matrix[0,0])
        abs_sin = abs(rotation_matrix[0,1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += bound_w/2 - image_center[0]
        rotation_matrix[1, 2] += bound_h/2 - image_center[1]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))
        rotated_images.append(rotated_image)

    return rotated_images


def crop_rectangle(
    image: np.ndarray,
    rects: np.ndarray,
) -> List[np.ndarray]:
    """rect has to be upright.

    Parameters
    ----------
    image: np.ndarray
        Image to be rotate and crop.

    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Rotat and crop rectangle.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    Returns
    -------
    croped_images: List[np.ndarray]
        Image after cropping.
    """
    croped_images = []
    height = image.shape[0]
    width = image.shape[1]

    # Determine if rect is inside the entire image
    inside_or_outsides = is_inside_rect(
        rects=rects,
        width_of_outer_rect=width,
        height_of_outer_rect=height,
    )

    rects = rects[inside_or_outsides, ...]

    for rect in rects:
        cx = int(rect[0])
        cy = int(rect[1])
        rect_width = int(rect[2])
        rect_height = int(rect[3])

        croped_image = image[
            cy-rect_height//2:cy+rect_height-rect_height//2,
            cx-rect_width//2:cx+rect_width-rect_width//2,
        ]
        croped_images.append(croped_image)

    return croped_images


def rotate_and_crop_rectangle(
    image: np.ndarray,
    rects_tmp: np.ndarray,
    operation_when_cropping_out_of_range: str,
) -> List[np.ndarray]:
    """Crop a rotated rectangle from a image.

    Parameters
    ----------
    image: np.ndarray
        Image to be rotate and crop.

    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Rotat and crop rectangle.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    operation_when_cropping_out_of_range: str
        'padding' or 'ignore'

    Returns
    -------
    rotated_croped_image: List[np.ndarray]
        Image after cropping and rotation.
    """
    rects = copy.deepcopy(rects_tmp)
    rotated_croped_images = []
    height = image.shape[0]
    width = image.shape[1]

    # Determine if rect is inside the entire image
    if operation_when_cropping_out_of_range == 'padding':
        size = (int(math.sqrt(width ** 2 + height ** 2)) + 2) * 2
        image = pad_image(
            image=image,
            resize_width=size,
            resize_height=size,
        )
        rects[:, 0] = rects[:, 0] + abs(size-width) / 2
        rects[:, 1] = rects[:, 1] + abs(size-height) / 2

    elif operation_when_cropping_out_of_range == 'ignore':
        inside_or_outsides = is_inside_rect(
            rects=rects,
            width_of_outer_rect=width,
            height_of_outer_rect=height,
        )
        rects = rects[inside_or_outsides, ...]

    rect_bbx_upright = bounding_box_from_rotated_rect(
        rects=rects,
    )

    rect_bbx_upright_images = crop_rectangle(
        image=image,
        rects=rect_bbx_upright,
    )

    rotated_rect_bbx_upright_images = image_rotation_without_crop(
        images=rect_bbx_upright_images,
        angles=rects[..., 4:5],
    )

    for rotated_rect_bbx_upright_image, rect in zip(rotated_rect_bbx_upright_images, rects):
        crop_cx = rotated_rect_bbx_upright_image.shape[1]//2
        crop_cy = rotated_rect_bbx_upright_image.shape[0]//2
        rect_width = int(rect[2])
        rect_height = int(rect[3])

        rotated_croped_images.append(
            rotated_rect_bbx_upright_image[
                crop_cy-rect_height//2:crop_cy+(rect_height-rect_height//2),
                crop_cx-rect_width//2:crop_cx+(rect_width-rect_width//2),
            ]
        )

    return rotated_croped_images


def keep_aspect_resize_and_pad(
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize the image based on the long side to fit inside
    the specified bounding rectangle while maintaining the aspect ratio,
    and padding the short side to the specified size.

    Parameters
    ----------
    image: np.ndarray
        Image to be resize and pad.

    resize_width: int
        Width of outer rectangle.

    resize_width: int
        Height of outer rectangle

    Returns
    -------
    padded_image: np.ndarray
        Image after padding.

    resized_image: np.ndarray
        Image after resize. (Before padding)
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    padded_image = np.zeros(
        (resize_height, resize_width, 3),
        np.uint8
    )
    ash = resize_height / image_height
    asw = resize_width / image_width
    if asw < ash:
        sizeas = (int(image_width * asw), int(image_height * asw))
    else:
        sizeas = (int(image_width * ash), int(image_height * ash))
    resized_image = cv2.resize(image, dsize=sizeas)
    start_h = int(resize_height / 2 - sizeas[1] / 2)
    end_h = int(resize_height / 2 + sizeas[1] / 2)
    start_w = int(resize_width / 2 - sizeas[0] / 2)
    end_w = int(resize_width / 2 + sizeas[0] / 2)
    padded_image[start_h:end_h, start_w:end_w, :] = resized_image.copy()

    return padded_image, resized_image


def pad_image(
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> np.ndarray:
    """Padding the perimeter of the image to the specified bounding rectangle size.

    Parameters
    ----------
    image: np.ndarray
        Image to be resize and pad.

    resize_width: int
        Width of outer rectangle.

    resize_width: int
        Height of outer rectangle

    Returns
    -------
    padded_image: np.ndarray
        Image after padding.
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    if resize_width < image_width:
        resize_width = image_width
    if resize_height < image_height:
        resize_height = image_height

    padded_image = np.zeros(
        (resize_height, resize_width, 3),
        np.uint8
    )
    start_h = int(resize_height / 2 - image_height / 2)
    end_h = int(resize_height / 2 + image_height / 2)
    start_w = int(resize_width / 2 - image_width / 2)
    end_w = int(resize_width / 2 + image_width / 2)
    padded_image[start_h:end_h, start_w:end_w, :] = image

    return padded_image


def rotated_rect_to_points(
    cx: float,
    cy: float,
    width: float,
    height: float,
    rotation: float,
) -> List[List[int]]:
    """Obtain coordinates for drawing a rotated rectangle

    Parameters
    ----------
    cx: float
        Rectangle center X coordinate.

    cy: float
        Rectangle center Y coordinate.

    width: float
        Width of the rectangle.

    height: float
        Height of the rectangle.

    rotation: float
        The rotation radians in a clockwise direction.

    Returns
    -------
    rotated_croped_image: List[np.ndarray]
        4 coordinates for drawing rotated rectangle.

        e.g.\n
        cv2.polylines(\n
            img=frame,\n
            pts=[np.array([[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]])],\n
            isClosed=True,\n
            color=(0,255,255),\n
            thickness=2,\n
            lineType=cv2.LINE_AA,\n
        )
    """
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    p0x = cx - a*height - b*width
    p0y = cy + b*height - a*width
    p1x = cx + a*height - b*width
    p1y = cy - b*height - a*width
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)

    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]
