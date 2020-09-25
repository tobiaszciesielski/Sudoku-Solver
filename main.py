import cv2
import operator
import numpy as np
import pickle
import time
import copy


RED_COLOR = (0, 0, 255)
BLACK_COLOR = (0, 0, 0)

sudoku = list()
sudoku_copy = list()



######################################
# iterative sudoku solving method v1 #
######################################
def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if arr[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False


def used_in_row(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False


def used_in_col(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False


def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[i + row][j + col] == num:
                return True
    return False


def check_location_is_safe(arr, row, col, num):
    return not used_in_row(arr, row, num) and not used_in_col(arr, col, num) and not used_in_box(arr, row - row % 3,
                                                                                                 col - col % 3, num)

def solve_sudoku(arr):
    global sudoku

    l = [0, 0]

    if not find_empty_location(arr, l):
        return True

    row = l[0]
    col = l[1]

    for num in range(1, 10):
        if check_location_is_safe(arr, row, col, num):
            arr[row][col] = num
            if solve_sudoku(arr):
                sudoku = arr
                return True
            arr[row][col] = 0
    return False


######################################
# recursive sudoku solving method v1 #
######################################
# def is_possible(y, x, n):
#     global sudoku
#     for i in range(9):
#         if sudoku[y][i] == n:
#             return False
#
#     for i in range(9):
#         if sudoku[i][x] == n:
#             return False
#
#     y_sq, x_sq = (y // 3) * 3, (x // 3) * 3
#     for i in range(y_sq, y_sq + 3):
#         for j in range(x_sq, x_sq + 3):
#             if sudoku[i][j] == n:
#                 return False
#     return True

#
# def solve():
#     global sudoku
#     global sudoku_copy
#     for y in range(9):
#         for x in range(9):
#             if sudoku[y][x] == 0:
#                 for n in range(1, 10):
#                     if is_possible(y, x, n):
#                         sudoku[y][x] = n
#                         solve()
#                         sudoku[y][x] = 0
#                 return
#     print(np.matrix(sudoku))


def load_svm_model():
    with open('svm_model.pkl', 'rb') as file:
        return pickle.load(file)


def center_image(image_to_process):
    OFFSET = 18, 7
    # X1, X2, Y1, Y2 = 15, 47, 7, 56 # base area
    X1, X2, Y1, Y2 = 23, 49, 7, 56

    contours, hierarchy = cv2.findContours(image_to_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.array([pt[0] for pt in contours[0]])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    cropped = image_to_process[y1:y2, x1:x2]
    cropped = cv2.resize(cropped, (X2-X1, Y2-Y1), interpolation=cv2.INTER_AREA)
    dst_image = np.zeros((64, 64))
    dst_image[OFFSET[1]:OFFSET[1]+(Y2-Y1), OFFSET[0]:OFFSET[0]+(X2-X1)] = cropped
    return dst_image


def find_biggest_grid(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    offset = 3
    top_left = top_left[0]+offset, top_left[1]+offset
    top_right = top_right[0]-offset, top_right[1]+offset
    bottom_right = bottom_right[0]-offset, bottom_right[1]-offset
    bottom_left = bottom_left[0]+offset, bottom_left[1]-offset

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    dst = np.array([[0, 0], [576, 0], [576, 576], [0, 576]], dtype='float32')
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (576, 576))


def find_sudoku_grid(image):
    dst = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dst = cv2.bilateralFilter(dst, 9, 75, 75)
    dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    sudoku_contours = find_biggest_grid(dst)
    sudoku_grid = crop_and_warp(image, sudoku_contours)
    return sudoku_grid


def process_sudoku_grid(sudoku_grid, model):
    sudoku_grid = cv2.cvtColor(sudoku_grid, cv2.COLOR_RGB2GRAY)
    _, sudoku_grid = cv2.threshold(sudoku_grid, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rows = []
    for k in range(9):
        top_left = [(i*64, k*64) for i in range(9)]
        bottom_right = [(j*64, k*64+64) for j in range(1, 10)]
        rows.append((top_left, bottom_right))

    sudoku_numbers = []
    for j in range(9):
        row = []
        for i in range(9):
            x1, y1 = rows[j][0][i]
            x2, y2 = rows[j][1][i]

            image_2_predict = sudoku_grid[y1+10:y2-10, x1+10:x2-10]
            n_white_pix = np.sum(image_2_predict == 255)
            if n_white_pix > 120:
                image_2_predict = center_image(image_2_predict)
                y = model.predict(np.array([image_2_predict.reshape(-1)]))[0]
                row.append(y)
            else:
                row.append(0)
        sudoku_numbers.append(row)
    return sudoku_numbers


def add_to_pixels(image, value):
    image = np.array(image, dtype=np.float32)
    image[:, :] += value
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    return image


def place_hint_on_screen(image, perc=62.5):
    y, x, ch = image.shape
    background = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = add_to_pixels(background, -100)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    square_size = int(x * (perc/100))  # x * perc of x dimension. Example: for 640x480 it is 400x400
    if square_size > y: square_size = y
    offset = (x - square_size) // 2, (y - square_size) // 2
    x1, y1, x2, y2 = offset[0], offset[1], offset[0] + square_size, offset[1] + square_size
    background = cv2.rectangle(background, (x1, y1), (x2, y2),
                               color=(255, 0, 0), thickness=3)
    background[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return background, ((x1, y1), (x2, y2))


def place_text(image, text="", origin=(0, 0), color=(0, 0, 0), thickness=1, size=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, text, origin, font,
                        fontScale=size,
                        color=color,
                        thickness=thickness,
                        lineType=cv2.LINE_AA)
    return image


def draw_solution(output, cpy):
    global sudoku
    rows = []
    for k in range(9):
        top_left = [(i * 64, k * 64) for i in range(9)]
        bottom_right = [(j * 64, k * 64 + 64) for j in range(1, 10)]
        rows.append((top_left, bottom_right))

    for j in range(9):
        for i in range(9):
            pt = rows[j][0][i][0] + 18, rows[j][0][i][1] + 36
            if cpy[j][i] == 0:
                place_text(output, str(sudoku[j][i]), pt, BLACK_COLOR, 3, 1)


def sudoku_solver(sudoku_image):
    svm_model = load_svm_model()
    sudoku_grid = find_sudoku_grid(sudoku_image)
    sudoku = process_sudoku_grid(sudoku_grid, svm_model)
    cpy = copy.deepcopy(sudoku)
    if solve_sudoku(sudoku):
        draw_solution(sudoku_grid, cpy)
        sudoku_grid = cv2.resize(sudoku_grid, (480,480), interpolation=cv2.INTER_AREA)
        cv2.imshow('Sudoku solver', sudoku_grid)
        cv2.waitKey()
    else:
        print('error')



def main():
    cap = cv2.VideoCapture()
    cap.open(0)
    cv2.namedWindow('Sudoku solver')
    waiting = False
    photo_taken = False
    while True:
        if not photo_taken:
            _, frame = cap.read()
            frame, square_pts = place_hint_on_screen(frame, perc=65)

            #  Check key events
            key_pressed = cv2.waitKey(33)
            if key_pressed == 32 and (not waiting):  # ' ' take picture
                waiting = True
                start = time.time()
            elif key_pressed == 113:  # 'q' quit
                break

            #  wait 3 seconds
            if waiting:
                elapsed = (time.time() - start) % 60
                text = str(3 - int(elapsed))
                if elapsed > 3:
                    text = ''
                    waiting = False
                    photo_taken = True
            else:
                text = "To take photo, press SPACE!"

            frame = place_text(frame, text, (square_pts[0][0] + 30, square_pts[0][1] - 10), RED_COLOR, 1, 0.8)
            cv2.imshow('Sudoku solver', frame)
        elif photo_taken is True:
            text = "IS IT OK? Y / N"
            key_pressed = cv2.waitKey(33)
            if key_pressed == 121:  # 'y'  lets start process !
                x1, y1, x2, y2 = square_pts[0][0], square_pts[0][1], square_pts[1][0], square_pts[1][1]
                sudoku_solver(frame[y1:y2, x1:x2])
                photo_taken = False
            elif key_pressed == 110:  # 'n' take pic again
                photo_taken = False
            elif key_pressed == 113:  # 'q' quit
                break
            frame = place_text(frame, text, (square_pts[0][0] + 30, square_pts[0][1] - 10), RED_COLOR, 1, 0.8)
            cv2.imshow('Sudoku solver', frame)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
