IMG_WIDTH = 416
IMG_HEIGHT = 416
CENTER_POS_MIN=60
CENTER_POS_MAX=360
RECT_LENGTH_MIN=32
RECT_LENGTH_MAX=128
CEL_NUMS=13
CEL_LEN=int(IMG_WIDTH/CEL_NUMS)
CLASSIC_NUMS=20

def get_two_line_cross_line(l1,l2):
    if l2[1]<l1[0]:
        return None
    if l2[0]>l1[1]:
        return None
    min_x = max(l1[0],l2[0])
    max_x = min(l1[1],l2[1])
    return [min_x,max_x]

def get_two_rect_cross_rect(rect1,rect2):
    x_range = get_two_line_cross_line([rect1[0],rect1[2]],[rect2[0],rect2[2]])
    if x_range == None:
        return None
    y_range = get_two_line_cross_line([rect1[1],rect1[3]],[rect2[1],rect2[3]])
    if y_range == None:
        return None
    return [x_range[0],y_range[0],x_range[1],y_range[1]]

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

if __name__=='__main__':
    rect1 = [10,10,20,20]
    rect2 = [21,5,21,19]
    data = get_two_rect_cross_rect(rect1,rect2)
    print(data)