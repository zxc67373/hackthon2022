import os
import numpy as np



# prediction
prediction_p = os.path.join('A.txt')
# gt
gt_p = os.path.join('test_target(1).txt')

def get_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max((ix2 - ix1), 0) * max((iy2 - iy1), 0)
    return inter / (area1 + area2 - inter)


# 取数据
def get_data(p):
    data = {}
    with open(p) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            line = line.split(',')
            if line[0] not in data.keys():
                data[line[0]] = []
            data[line[0]].append([int(x) for x in line[1:]])

    return data


# 取预测数据
predict_data = get_data(prediction_p)
# 取真实数据
gt_data = get_data(gt_p)

# 记录精度用，依次是高处作业带安全带、高处作业未带安全带
tp = [0, 0]
t = [0, 0]
p = [0, 0]

c = 0
for key in gt_data.keys():
    data = gt_data[key]
    c += 1
    boxes = []
    # 先记录正例数量
    for d in data:
        d = [int(x) for x in d]
        boxes.append(d)
        # 双条件
        if d[4:6] == [1, 1]:
            t[0] += 1
        if d[4:6] == [1, 0]:
            t[1] += 1

    # 如果预测没有该图的数据跳过
    if key not in predict_data.keys():
        continue

    # 该图预测的结果
    pre_data = predict_data[key]
    # 用于存储预测框和真实框的iou
    iou_map = np.zeros((len(pre_data), len(boxes)))

    for i in range(len(pre_data)):
        p_d = pre_data[i]
        p_d = [int(x) for x in p_d]
        # 记录预测的p的结果
        if p_d[4:6] == [1, 1]:
            p[0] += 1
        if p_d[4:6] == [1, 0]:
            p[1] += 1

        p_box = p_d[:4]
        # 计算下iou_map
        for j in range(len(boxes)):
            gt_box = boxes[j]
            iou_score = get_iou(p_box, gt_box[:4])
            if iou_score < 0.5:
                iou_score = 0
            iou_map[i][j] = iou_score
    # 记录了预测框对应的最大iou真实框的下标
    iou_max_index = iou_map.argmax(axis=1)
    # 标记已匹配gt框
    flag = [0 for _ in range(len(boxes))]
    for i in range(len(pre_data)):
        # 与第i个预测框iou最大的真实框下标
        j = iou_max_index[i]
        # iou分数
        ij_score = iou_map[i][j]
        # 小于0.5不计算tp
        if ij_score < 0.5:
            continue
        # iou>=0.5,该gt已被匹配过,统计的p减1，且不计算tp
        if flag[j] == 1:
            if pre_data[i][4:6] == [1, 1]:
                p[0] -= 1
            if pre_data[i][4:6] == [1, 0]:
                p[1] -= 1
            continue

        # 标记为匹配过
        flag[j] = 1
        # pre_d = [int(x) for x in pre_data[i]]
        pre_d = pre_data[i]
        if pre_d[4:6] == boxes[j][4:6]:
            if pre_d[4:6] == [1, 1]:
                tp[0] += 1
            if pre_d[4:6] == [1, 0]:
                tp[1] += 1

    # print(iou_map, iou_max_index)

print(t, p, tp)
t = [float(x) for x in t]
p = [float(x) for x in p]
tp = [float(x) for x in tp]
precision = [0, 0]
recall = [0, 0]
f1 = [0, 0]
for i in range(2):
    precision[i] = tp[i] / p[i]
    recall[i] = tp[i] / t[i]
    f1[i] = 2 * (precision[i] * recall[i]) / float(precision[i] + recall[i] + 1e-8)

print(f1)

final_score = f1[0] * 0.5 + f1[1] * 0.5
print(final_score)