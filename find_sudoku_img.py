import cv2
import numpy as np
import easyocr


def open_image(filename):
    ''' Открываем изображение с судоку '''
    return cv2.imread(filename)


def find_contours(img):
    ''' Находим контуры на изображении '''

    # Делаем изображение черно-белым
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Добавляем небольшое размытие
    blurry = cv2.GaussianBlur(gray, (5, 5), 5)
    # Отсеиваем полу-прозрачные цвета
    thresh = cv2.adaptiveThreshold(blurry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
    # Ищем контуры
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    return cnts


def find_biggest_square(cnts):
    ''' Для поиска сетки судоку находим наибольший квадрат на изображении '''

    location = None
    for cnt in cnts:
        # Сглаживаем неровности в контуре
        approx = cv2.approxPolyDP(cnt, 15, True)
        # Если точки 4 - квардрат
        if len(approx) == 4:

            # Сортировка углов по часовой стрелке
            rect = np.zeros((4, 2), dtype="float32")
            cutt = approx[:, 0]

            diag_1 = cutt.sum(axis=1)
            rect[0] = cutt[np.argmin(diag_1)]
            rect[2] = cutt[np.argmax(diag_1)]

            diag_2 = np.diff(cutt, axis=1)
            rect[1] = cutt[np.argmin(diag_2)]
            rect[3] = cutt[np.argmax(diag_2)]

            location = rect
            break
    return location


def print_sudoku(img, location):
    height = 900
    width = 900
    pts1 = np.float32([location[0], location[1], location[3], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Вписываем судоку в наш квадрат
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    board = cv2.warpPerspective(img, matrix, (width, height))
    # Обрабатываем для лучшей читаемости
    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    board = cv2.threshold(board, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # show_image(board)
    return board


def read_sudoku(board):
    reader = easyocr.Reader(['en'], gpu=True)

    # Создаём датафрейм и список для записи распознанных результатов
    sudoku_map = []

    # Разделяем наш судоку на 9 строк и 9 столбцов и распознаём каждое значение
    split = np.split(board, 9)
    for row_index, row_img in enumerate(split):
        digs = np.split(row_img, 9, axis=1)
        sudoku_map.append(list())
        for col_img in digs:
            # Убираем рамки
            col_img = col_img[10:90, 10:90]
            # col_img = cv2.blur(col_img, (5, 5))

            # Распознаём число в ячейке и записываем его в датафрейм и список с координатами
            text = reader.readtext(col_img, detail=0, mag_ratio=2)
            # print(text)
            # show_image(col_img)
            sudoku_map[row_index].append(next(iter(text), ''))

    return sudoku_map


def show_image(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)
    # while True:
    #     k = cv2.waitKey(0) & 0xFF
    #     if k == 27:
    #         cv2.destroyAllWindows()
    #         break


def extract_sudoku(filename):
    img = open_image(filename)
    cnts = find_contours(img)
    location = find_biggest_square(cnts)
    board = print_sudoku(img, location)
    sudoku_map = read_sudoku(board)
    return sudoku_map


if __name__ == "__main__":
    from pprint import pprint

    img = open_image('test_img3.jpg')
    cnts = find_contours(img)
    location = find_biggest_square(cnts)
    board = print_sudoku(img, location)
    sudoku_map = read_sudoku(board)
    pprint(sudoku_map)
    # cv2.drawContours(img, [location], -1, (0, 0, 255), 3)
