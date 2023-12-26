import cv2
import numpy as np
from sklearn.cluster import KMeans

drawing = False
ix, iy = -1, -1
x1, y1, x2, y2 = -1, -1, -1, -1

def get_green_ratio(image, x1, y1, x2, y2):
    selected_region = image[y1:y2, x1:x2]
    green_color = [0, 255, 0]
    green_mask = np.all(selected_region == green_color, axis=2)
    total_pixels = (x2 - x1) * (y2 - y1)
    green_pixels = np.count_nonzero(green_mask)
    green_ratio = green_pixels / total_pixels
    return green_ratio

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, x1, y1, x2, y2, copy, selected_region
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            copy = image.copy()
            cv2.rectangle(copy, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        cv2.rectangle(copy, (ix, iy), (x, y), (0, 255, 0), 2)
        x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
        selected_region = image[y1:y2, x1:x2]

def get_dominant_color(image, x1, y1, x2, y2, k=3):
    selected_region = image[y1:y2, x1:x2]
    selected_region = selected_region.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(selected_region)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    dominant_color_counts = np.bincount(kmeans.labels_)
    dominant_color_index = np.argmax(dominant_color_counts)
    dominant_color = dominant_colors[dominant_color_index]
    return dominant_color

image_path = '1.jpg'
image = cv2.imread(image_path)
copy = image.copy()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_rectangle)

while True:
    cv2.imshow('Image', copy)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and selected_region is not None:
        cv2.imwrite('selected_region.jpg', selected_region)
        break

selected_region = image[y1:y2, x1:x2]
green_lower = np.array([35, 50, 50], dtype="uint8")
green_upper = np.array([90, 255, 255], dtype="uint8")
hsv_image = cv2.cvtColor(selected_region, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_image, green_lower, green_upper)
green_pixels = cv2.countNonZero(mask)
total_pixels = (x2 - x1) * (y2 - y1)
green_ratio = (green_pixels / total_pixels) * 100
print(f"Лягушка зелена на {green_ratio} %")
print(f"Лягушка зелена (RGB) на {green_pixels / 255} %")

selected_region_reshaped = selected_region.reshape((-1, 3))
kmeans = KMeans(n_clusters=1).fit(selected_region_reshaped)
dominant_color = kmeans.cluster_centers_[0]
print("Лягушка определена с цветом (RGB):", dominant_color)

cv2.destroyAllWindows()
