'''
  @ Date: 2021-04-13 16:14:36
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-25 20:56:26
  @ FilePath: /EasyMocapRelease/easymocap/annotator/chessboard.py
'''
import numpy as np
import cv2
from func_timeout import func_set_timeout

# Keys must match printed board / CharucoBoard / CLI --aruco_type
ARUCO_PREDEFINED = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}


def _build_charuco_board(long, short, squareLength, aruco_len, dictionary):
    """OpenCV 4.7+ removed CharucoBoard_create; use CharucoBoard((squaresX, squaresY), ...)."""
    ar = cv2.aruco
    if hasattr(ar, "CharucoBoard_create"):
        return ar.CharucoBoard_create(
            squaresY=long,
            squaresX=short,
            squareLength=squareLength,
            markerLength=aruco_len,
            dictionary=dictionary,
        )
    return ar.CharucoBoard((short, long), squareLength, aruco_len, dictionary)


def _charuco_board_corners_xyz(board):
    """3D chessboard corner coordinates in board frame (legacy property or OpenCV 4.x getter)."""
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    else:
        corners = board.chessboardCorners
    return np.asarray(corners, dtype=np.float64)


def _aruco_detect_markers(image, dictionary, parameters=None, detector=None):
    """
    OpenCV 4.7+ removed cv2.aruco.detectMarkers; use ArucoDetector.detectMarkers instead.
    Pass ``detector`` if already built (e.g. CharucoBoard fallback path).
    """
    ar = cv2.aruco
    if detector is not None:
        return detector.detectMarkers(image)
    if parameters is None:
        parameters = ar.DetectorParameters()
    if hasattr(ar, "detectMarkers"):
        return ar.detectMarkers(image=image, dictionary=dictionary, parameters=parameters)
    if not hasattr(ar, "ArucoDetector"):
        raise AttributeError("cv2.aruco has neither detectMarkers nor ArucoDetector")
    return ar.ArucoDetector(dictionary, parameters).detectMarkers(image)


def _aruco_interpolate_corners_charuco(marker_corners, marker_ids, image, board):
    ar = cv2.aruco
    if not hasattr(ar, "interpolateCornersCharuco"):
        return False, None, None
    return ar.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=image,
        board=board,
    )


def getChessboard3d(pattern, gridSize, axis='xy'):
    object_points = np.zeros((pattern[1]*pattern[0], 3), np.float32)
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    object_points[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    object_points[:, [0, 1]] = object_points[:, [1, 0]]
    object_points = object_points * gridSize
    if axis == 'zx':
        object_points = object_points[:, [1, 2, 0]]
    return object_points

colors_chessboard_bar = [
    [0, 0, 255],
    [0, 128, 255],
    [0, 200, 200],
    [0, 255, 0],
    [200, 200, 0],
    [255, 0, 0],
    [255, 0, 250]
]

def get_lines_chessboard(pattern=(9, 6)):
    w, h = pattern[0], pattern[1]
    lines = []
    lines_cols = []
    for i in range(w*h-1):
        lines.append([i, i+1])
        lines_cols.append(colors_chessboard_bar[(i//w)%len(colors_chessboard_bar)])
    return lines, lines_cols

def _findChessboardCorners(img, pattern, debug, fix_orientation):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    if fix_orientation:
        h, w = pattern[1], pattern[0]
        corners = corners.reshape(h, w, 2)

        # horizontal flip
        if corners[0, -1, 0] < corners[0, 0, 0]:
            corners = corners[:, ::-1]

        # vertical flip
        if corners[-1, 0, 1] < corners[0, 0, 1]:
            corners = corners[::-1, :]

        corners = corners.reshape(-1, 2)
    else:
        corners = corners.squeeze()

    return True, corners

def _findChessboardCornersAdapt(img, pattern, debug, fix_orientation):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 21, 2)
    return _findChessboardCorners(img, pattern, debug, fix_orientation)

@func_set_timeout(5)
def findChessboardCorners(img, annots, pattern, debug=False, fix_orientation=False):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    for func in [_findChessboardCorners, _findChessboardCornersAdapt]:
        ret, corners = func(gray, pattern, debug, fix_orientation)
        if ret:break
    else:
        return None
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show

def create_chessboard(path, keypoints3d, out='annots'):
    from tqdm import tqdm
    from os.path import join
    from .file_utils import getFileList, save_json, read_json
    import os
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(join(path, 'images'), ext='.jpg', max=1)
    imgnames = [join('images', i) for i in imgnames]
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', out).replace('.jpg', '.json')
        annname = join(path, annname)
        if not os.path.exists(annname):
            save_json(annname, template)
        elif True:
            annots = read_json(annname)
            annots['keypoints3d'] = template['keypoints3d']
            save_json(annname, annots)


def detect_charuco(image, aruco_type, long, short, squareLength, aruco_len):
    dictionary = cv2.aruco.getPredefinedDictionary(dict=ARUCO_PREDEFINED[aruco_type])
    board = _build_charuco_board(long, short, squareLength, aruco_len, dictionary)
    corners = _charuco_board_corners_xyz(board)
    # ATTN: exchange the XY
    corners3d = corners[:, [1, 0, 2]]
    keypoints2d = np.zeros_like(corners3d)
    # 查找标志块的左上角点
    corners, ids, _ = _aruco_detect_markers(image, dictionary)
    # 棋盘格黑白块内角点
    if ids is not None:
        retval, charucoCorners, charucoIds = _aruco_interpolate_corners_charuco(
            corners, ids, image, board
        )
        if retval:
            ids = charucoIds[:, 0]
            pts = charucoCorners[:, 0]
            keypoints2d[ids, :2] = pts
            keypoints2d[ids, 2] = 1.
    else:
        retval = False
    return retval, keypoints2d, corners3d

class CharucoBoard:
    def __init__(self, long, short, squareLength, aruco_len, aruco_type) -> None:
        '''
            long, short: 沿 Y / X 方向的方格个数（OpenCV: squaresY, squaresX），不是内角点个数。
            内角点网格尺寸为 (long-1) x (short-1)，与 template['pattern'] 一致。
            squareLength, aruco_len: 方格边长与 ArUco 码边长（米，须与印刷板一致）。
            aruco_type: 见 ARUCO_PREDEFINED。
        '''
        if aruco_type not in ARUCO_PREDEFINED:
            raise KeyError(f'Unknown aruco_type {aruco_type!r}; use one of {list(ARUCO_PREDEFINED)}')
        dictionary = cv2.aruco.getPredefinedDictionary(dict=ARUCO_PREDEFINED[aruco_type])
        board = _build_charuco_board(long, short, squareLength, aruco_len, dictionary)
        corners = _charuco_board_corners_xyz(board)
        # ATTN: exchange the XY
        corners = corners[:, [1, 0, 2]]
        self.template = {
            'keypoints3d': corners,
            'keypoints2d': np.zeros_like(corners),
            'pattern': (long - 1, short - 1),
            'grid_size': squareLength,
            'visited': False
        }
        self.long = long
        self.short = short
        self.squareLength = squareLength
        self.aruco_len = aruco_len
        self.aruco_type = aruco_type
        self.dictionary = dictionary
        self.board = board
        self._aruco_marker_detector = None
        if not hasattr(cv2.aruco, "detectMarkers") and hasattr(cv2.aruco, "ArucoDetector"):
            try:
                self._aruco_marker_detector = cv2.aruco.ArucoDetector(
                    self.dictionary, cv2.aruco.DetectorParameters()
                )
            except Exception:
                self._aruco_marker_detector = None
        self._charuco_detector = None
        # CharucoDetector needs cv2.aruco.CharucoBoard (new API); duplicate only if legacy create exists
        if hasattr(cv2.aruco, "CharucoDetector"):
            try:
                if hasattr(cv2.aruco, "CharucoBoard_create"):
                    det_board = cv2.aruco.CharucoBoard(
                        (short, long), squareLength, aruco_len, dictionary
                    )
                else:
                    det_board = self.board
                charuco_params = cv2.aruco.CharucoParameters()
                detector_params = cv2.aruco.DetectorParameters()
                refine_params = cv2.aruco.RefineParameters()
                self._charuco_detector = cv2.aruco.CharucoDetector(
                    det_board, charuco_params, detector_params, refine_params
                )
            except Exception:
                self._charuco_detector = None

    def detect(self, img_color, annots):
        """Fill annots['keypoints2d'] (numpy rows) for detected charuco corners; draw on img_color."""
        k2d = np.asarray(annots['keypoints2d'], dtype=np.float32)
        n = k2d.shape[0]
        k2d[:, :] = 0.0

        ok = False
        if self._charuco_detector is not None:
            charuco_corners, charuco_ids, _, _ = self._charuco_detector.detectBoard(img_color)
            if charuco_corners is not None and charuco_ids is not None:
                if len(charuco_corners) > 0:
                    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
                    pts = np.array(
                        [np.asarray(c, dtype=np.float32).reshape(-1)[:2] for c in charuco_corners],
                        dtype=np.float32,
                    )
                    if len(ids) == len(pts):
                        for i, cid in enumerate(ids):
                            if 0 <= cid < n:
                                k2d[cid, :2] = pts[i]
                                k2d[cid, 2] = 1.0
                        ok = k2d[:, 2].sum() > 0.1
                        if ok:
                            cv2.aruco.drawDetectedCornersCharuco(
                                img_color, charuco_corners, charuco_ids, cornerColor=(0, 0, 255)
                            )
        if not ok:
            corners, ids, _ = _aruco_detect_markers(
                img_color, self.dictionary, detector=self._aruco_marker_detector
            )
            if ids is not None:
                retval, charucoCorners, charucoIds = _aruco_interpolate_corners_charuco(
                    corners, ids, img_color, self.board
                )
            else:
                retval = False
            if retval:
                cv2.aruco.drawDetectedCornersCharuco(
                    img_color, charucoCorners, charucoIds, [0, 0, 255]
                )
                ids = charucoIds[:, 0]
                pts = charucoCorners[:, 0]
                k2d[ids, :2] = pts
                k2d[ids, 2] = 1.0
                ok = True
        annots['keypoints2d'] = k2d
        return ok

    def __call__(self, imgname, images='images', output='output'):
        import os
        from .file_utils import read_json, save_json
        import copy
        img_color = cv2.imread(imgname)
        annotname = imgname.replace('images', 'chessboard').replace('.jpg', '.json')
        if os.path.exists(annotname):
            annots = read_json(annotname)
            if annots['visited']:
                return
        else:
            annots = copy.deepcopy(self.template)
        annots['visited'] = True
        self.detect(img_color, annots)
        annots['keypoints2d'] = annots['keypoints2d'].tolist()
        annots['keypoints3d'] = annots['keypoints3d'].tolist()
        save_json(annotname, annots)


@func_set_timeout(5)
def findCharucoCorners(img, annots, charuco_board, debug=False):
    """Like findChessboardCorners: updates annots, returns debug image or None."""
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    show = img.copy()
    ok = charuco_board.detect(show, annots)
    if not ok:
        return None
    k2d = annots['keypoints2d']
    annots['keypoints2d'] = k2d.tolist() if isinstance(k2d, np.ndarray) else k2d
    return show