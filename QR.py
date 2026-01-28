import cv2
from pyzbar.pyzbar import decode
import numpy as np

def preprocess_and_find_qr_contours(image_path):
    # Чтение изображения
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image ", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Преобразование в бинарное изображение
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow(r"Binary Image ", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_image = np.zeros_like(image)

    for contour in contours1:
        approx = cv2.approxPolyDP(contour, 7, True)
        if len(approx) == 4:  # Ищем 4-угольные контуры
            cv2.drawContours(result_image, [approx], 0, (255, 255, 255), 1)
    cv2.imshow("Contours binary", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Отображение контуров
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Ищем 4-угольные контуры
            cv2.drawContours(image, [approx], 0, (0, 0, 255), 3)

    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def decode_qr(image_path):
    # Чтение изображения
    image = cv2.imread(image_path)

    # Распознавание QR-кодов
    qr_codes = decode(image)
    decoded_data = []

    for qr in qr_codes:
        data = qr.data.decode('utf-8')
        decoded_data.append(data)
        # Рисуем рамку вокруг QR-кода
        points = qr.polygon
        if len(points) == 4:
            pts = np.array([[p.x, p.y] for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 0, 255), 3)

    cv2.imshow("Detected QR Codes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return decoded_data

# Основной блок программы

image_path = "images/qr1.png"
print("Этап 1: Предварительная обработка и поиск контуров")
preprocess_and_find_qr_contours(image_path)

print("Этап 2: Распознавание QR-кода")
decoded_info = decode_qr(image_path)
if decoded_info:
    print("Закодированная информация:", decoded_info)
else:
    print("QR-код не найден.")
