import cv2
import numpy as np


# Bu fonksiyonu sınıfın DIŞINA yaz
def sort_contours_by_lines(contours, line_threshold=10):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in bounding_boxes]

    lines = []
    used = [False] * len(centers)

    for i, (cx, cy) in enumerate(centers):
        if used[i]:
            continue
        line = [i]
        used[i] = True
        for j, (cx2, cy2) in enumerate(centers):
            if not used[j] and abs(cy - cy2) < line_threshold:
                line.append(j)
                used[j] = True
        lines.append(line)

    lines.sort(key=lambda line: centers[line[0]][1])

    sorted_contours = []
    for line in lines:
        line_contours = [contours[i] for i in line]
        line_boxes = [bounding_boxes[i] for i in line]
        line_sorted = [c for _, c in sorted(zip(line_boxes, line_contours), key=lambda b: b[0][0])]
        sorted_contours.extend(line_sorted)

    return sorted_contours
