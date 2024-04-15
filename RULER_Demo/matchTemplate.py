# ●TM_SQDIFF:计算平方不同,计算出来的值越小，越相关
# ●TM_CCORR: 计算相关性,计算出来的值越大，越相关
# ●TM_CCOEFF: 计算相关系数,计算出来的值越大,越相关
# ●TM_SQDIFF_NORMED:计算归一化平方不同,计算出来的值越接近0,越相关
# ●TM_CCORR_NORMED:计算归- -化相关性,计算出来的值越接近1,越相关
# ●TM_CCOEFF_NORMED:计算归-化相关系数,计算出来的值越接近1,越相关

import cv2
import numpy as np

# 读取输入图像和模板图像
input_image = cv2.imread('../img/ruler2.jpg')
template = cv2.imread('../img/ruler_template.jpg')

# 将输入图像转换为灰度图像
input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 使用模板匹配函数
result = cv2.matchTemplate(input_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# 设定阈值
threshold = 0.4

# 找到匹配位置
locations = np.where(result >= threshold)
best_match_loc = None

# 在找到的所有匹配位置中选择最佳匹配
if len(locations[0]) > 0:
    best_match_val = -1
    for loc in zip(*locations[::-1]):
        match_val = result[loc[1], loc[0]]
        if match_val > best_match_val:
            best_match_val = match_val
            best_match_loc = loc

    # 获取模板图像的宽度和高度
    template_width, template_height = template_gray.shape[::-1]

    # 根据最佳匹配位置获取最佳匹配结果的图像
    best_match_result = input_image[best_match_loc[1]:best_match_loc[1] + template_height,
                        best_match_loc[0]:best_match_loc[0] + template_width]
    # 显示结果
    cv2.imshow('Input Image', input_image)
    cv2.imshow('best_match_result Image', best_match_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


else:
    print("未找到匹配结果！")

