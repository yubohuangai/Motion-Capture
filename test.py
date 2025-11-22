import cv2
img = cv2.imread("/mnt/yubo/emily/extri_data/images/09/000000.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, corners = cv2.findChessboardCorners(gray, (9,6))
ret, corners = cv2.findChessboardCorners(
    gray, (9,6),
    flags=cv2.CALIB_CB_FAST_CHECK
)

print("Result:", ret)


#     flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK