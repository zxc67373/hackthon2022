def get_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max((ix2 - ix1), 0) * max((iy2 - iy1), 0)
    return inter / (area1 + area2 - inter)

print(get_iou([368,544,487,732],[374,540,484,739]))