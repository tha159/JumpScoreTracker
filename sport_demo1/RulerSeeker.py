import cv2
import numpy as np


class RulerSeeker:
    input_image = []
    ruler_template = []
    blanket_match_result = []
    blanket_coordinate = []
    only_red_blanket = []
    only_ruler_img = []
    ruler_lines = []
    real_ruler_lines = []
    isFindRuler = False
    startJumpArea = []
    long_real_ruler_lines = []
    ruler_right_area = []

    def __init__(self, input_image, ruler_template):
        self.input_image = cv2.imread(input_image)
        self.template = cv2.imread(ruler_template)
        self.match_template(area_threshold=0.3)
        self.find_only_red_blanket()
        self.find_only_ruler_img()
        self.ruler2ruler_lines()
        self.ruler_lines2real_ruler_lines()
        self.get_ruler_right_area()
        if len(self.real_ruler_lines) == 46:
            self.isFindRuler = True

    def showInfo(self):
        self.show_img('blanket_match_result', self.blanket_match_result)
        self.show_img('only_red_blanket', self.only_red_blanket)
        self.show_img('only_ruler_img', self.only_ruler_img)
        self.show_ruler_lines()
        self.draw_real_ruler_lines(self.input_image)
        print(f'ruler刻度线条数为：{len(self.ruler_lines)}')
        print(f'blanket_coordinate为：{self.blanket_coordinate}')

    def show_img(self, winname, image):
        cv2.imshow(winname, image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()

    def match_template(self, area_threshold):
        # 读取输入图片和模板

        input_image_copy = self.input_image.copy()
        template_copy = self.template.copy()
        # 灰度化输入图片和模板
        input_image_gray = cv2.cvtColor(input_image_copy, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_copy, cv2.COLOR_BGR2GRAY)

        # 使用模板匹配函数
        result = cv2.matchTemplate(input_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        # 找到匹配位置
        locations = np.where(result >= area_threshold)
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
            best_match_result = self.input_image[best_match_loc[1]:best_match_loc[1] + template_height,
                                best_match_loc[0]:best_match_loc[0] + template_width]
            self.blanket_coordinate = best_match_loc
            self.blanket_match_result = best_match_result


        else:
            print("未找到匹配结果！")
            self.blanket_match_result = None

    def find_only_red_blanket(self):
        image = self.blanket_match_result.copy()

        # 将图像从 BGR 转换为 HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围
        lower_red = np.array([0, 30, 40])
        upper_red = np.array([255, 100, 150])

        # 创建掩码，找到HSV图像中的红色区域
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # 在原始图像中根据掩码找到红色区域
        red_area = cv2.bitwise_and(image, image, mask=mask)

        red_area_gray = cv2.cvtColor(red_area, cv2.COLOR_BGR2GRAY)
        # 查找图像中的轮廓
        contours, hierarchy = cv2.findContours(red_area_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化最大轮廓和最大面积
        max_contour = None
        max_area = -1

        # 遍历所有轮廓
        for contour in contours:
            # 计算当前轮廓的面积
            area = cv2.contourArea(contour)

            # 如果当前轮廓的面积大于最大面积，则更新最大面积和对应的轮廓
            if area > max_area:
                max_area = area
                max_contour = contour

        # 在原始图像上绘制轮廓
        mask_forContour = np.zeros_like(image)
        cv2.drawContours(mask_forContour, [max_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
        # 将原始图像与掩码图像进行按位与操作
        onlyRedArea = cv2.bitwise_and(image, mask_forContour)

        self.only_red_blanket = onlyRedArea
        self.startJumpArea.extend(self.find_top_left_non_zero_color(onlyRedArea))

    def find_top_left_non_zero_color(self, only_red_blanket):
        height = len(only_red_blanket)
        width = len(only_red_blanket[0])

        top_point = None
        left_point = None

        for y in range(height):
            for x in range(width):
                if not np.array_equal(only_red_blanket[y][x], [0, 0, 0]):
                    if left_point is None or x < left_point[0]:
                        left_point = [x, y]
                    if top_point is None or y < top_point[1]:
                        top_point = [x, y]

        return [left_point, top_point]

    def find_only_ruler_img(self):
        image = self.only_red_blanket.copy()

        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 对灰度图像进行二值化处理
        ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        # 查找图像中的轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 定义面积阈值范围
        min_area = 20  # 最小面积
        max_area = 200  # 最大面积

        # 新建一个全黑图像，用于绘制保留的轮廓
        result = np.zeros_like(image)

        # 在新图像上绘制符合面积范围的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                cv2.drawContours(result, [contour], -1, (255, 255, 255), cv2.FILLED)

        # 将保留的轮廓作为掩码，从原始图像中提取对应区域
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        masked_image = cv2.bitwise_and(image, image, mask=result)

        self.only_ruler_img = masked_image

    def ruler2ruler_lines(self):
        image = self.only_ruler_img.copy()
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用LSD算法检测直线
        lsd = cv2.createLineSegmentDetector(0)  # 参数0表示默认LSD算法
        lsd_lines, _, _, _ = lsd.detect(gray)

        # 调整阈值以达到所需的过滤效果
        filtered_lines = self.filter_short_lines(lsd_lines, threshold=1.5)

        # 尺子刻度线共46条
        if len(filtered_lines) == 92:
            big_group = self.group_lines(filtered_lines)
            middle_lines = self.extract_middle_lines(big_group)

            new_lines = [np.squeeze(arr) for arr in middle_lines]

            self.ruler_lines = new_lines
            self.startJumpArea.append([new_lines[0][0], new_lines[0][1]])
            self.startJumpArea.append([new_lines[0][2], new_lines[0][3]])

    def group_lines(self, filtered_lines):
        # 根据中点的 x 位置对线段进行排序
        sorted_lines = sorted(filtered_lines, key=lambda line: (line[0][0] + line[0][2]) / 2)

        # 每两个划分为一组
        groups = [sorted_lines[i:i + 2] for i in range(0, len(sorted_lines), 2)]

        # 把划分完的小组放进一个大组里
        big_group = []
        for group in groups:
            big_group.append(group)

        return big_group

    def filter_short_lines(self, lines, threshold):
        filtered_lines = []
        lengths = [self.line_length(line) for line in lines]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        for line, length in zip(lines, lengths):
            if length >= mean_length - threshold * std_length:
                filtered_lines.append(line)

        return filtered_lines

    def line_length(self, line):
        x1, y1, x2, y2 = line[0]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def extract_middle_lines(self, big_group):
        middle_lines = []
        for group in big_group:
            # Extracting endpoints of the lines in the group
            x1_1, y1_1, x2_1, y2_1 = group[0][0]
            x1_2, y1_2, x2_2, y2_2 = group[1][0]
            # 计算第一条线段的最上面和最下面的顶点
            if y1_1 > y2_1:
                top_1 = (x1_1, y1_1)
                bottom_1 = (x2_1, y2_1)
            else:
                top_1 = (x2_1, y2_1)
                bottom_1 = (x1_1, y1_1)

            # 计算第二条线段的最上面和最下面的顶点
            if y1_2 > y2_2:
                top_2 = (x1_2, y1_2)
                bottom_2 = (x2_2, y2_2)
            else:
                top_2 = (x2_2, y2_2)
                bottom_2 = (x1_2, y1_2)

            # 计算顶点对的中点坐标
            midpoint_1 = ((top_1[0] + top_2[0]) / 2, (top_1[1] + top_2[1]) / 2)
            midpoint_2 = ((bottom_1[0] + bottom_2[0]) / 2, (bottom_1[1] + bottom_2[1]) / 2)

            # Creating a new line segment with the midpoint as both endpoints
            middle_line = np.array([[[midpoint_1[0], midpoint_1[1], midpoint_2[0], midpoint_2[1]]]], dtype=np.float32)
            middle_lines.append(middle_line)

        return middle_lines

    def show_ruler_lines(self):
        image = self.blanket_match_result.copy()
        # 在图像上绘制检测到的直线
        for line in enumerate(self.ruler_lines):
            x1, y1, x2, y2 = map(int, line[1])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1)  # 绘制直线
        self.show_img('ruler_lines', image)

    def ruler_lines2real_ruler_lines(self):
        if self.blanket_coordinate is None:
            print("未找到匹配结果！")
            return None

            # 获取匹配区域的坐标
        match_x, match_y = self.blanket_coordinate

        # 将ruler线条的坐标与匹配区域的坐标相结合
        real_ruler_lines = []
        for line in self.ruler_lines:
            # line 是一个元组 (x1, y1, x2, y2)，代表一条线的两个端点坐标
            x1, y1, x2, y2 = line
            # 将线条的坐标转换为相对于整个图像的坐标
            real_x1, real_y1 = x1 + match_x, y1 + match_y
            real_x2, real_y2 = x2 + match_x, y2 + match_y
            real_ruler_lines.append([[real_x1, real_y1], [real_x2, real_y2]])

        self.real_ruler_lines = real_ruler_lines

        p1, p2, p3, p4 = self.startJumpArea
        # 将线条的坐标转换为相对于整个图像的坐标
        real_x1, real_y1 = p1[0] + match_x, p1[1] + match_y
        real_x2, real_y2 = p2[0] + match_x, p2[1] + match_y
        real_x3, real_y3 = p3[0] + match_x, p3[1] + match_y
        real_x4, real_y4 = p4[0] + match_x, p4[1] + match_y

        self.startJumpArea = [[real_x1, real_y1], [real_x2, real_y2], [real_x3, real_y3], [real_x4, real_y4]]

        for real_ruler_line in self.real_ruler_lines:
            self.long_real_ruler_lines.append(self.extend_line(real_ruler_line))
        self.long_real_ruler_lines = self.convert_to_int(self.long_real_ruler_lines)


    def convert_to_int(self,lst):
        new_lst = []
        for item in lst:
            if isinstance(item, list):
                new_lst.append(self.convert_to_int(item))  # 递归处理嵌套列表
            else:
                new_lst.append(int(item))  # 转换为整数
        return new_lst

    def draw_real_ruler_lines(self, image):
        # 在图像上绘制检测到的直线
        for line in self.real_ruler_lines:
            x1, y1 = map(int, line[0])
            x2, y2 = map(int, line[1])
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1)  # 绘制直线

        # 将点的坐标转换为整数
        self.startJumpArea = self.sort_points_clockwise(self.startJumpArea)
        startJumpArea_points = np.array(self.startJumpArea, dtype=np.int32)

        # 绘制多边形
        cv2.polylines(image, [startJumpArea_points], isClosed=True, color=(0, 0, 255), thickness=1)

        # 将绘制的多边形叠加在原图像上
        cv2.addWeighted(image, 0.5, image, 0.5, 0, image)

    def draw_long_real_ruler_lines(self, image):
        # 在图像上绘制检测到的直线
        for line in self.long_real_ruler_lines:
            x1, y1 = map(int, line[0])
            x2, y2 = map(int, line[1])
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1)  # 绘制直线

        ##绘制起跳区域
        # 将点的坐标转换为整数
        self.startJumpArea = self.sort_points_clockwise(self.startJumpArea)
        startJumpArea_points = np.array(self.startJumpArea, dtype=np.int32)

        # 绘制多边形
        cv2.polylines(image, [startJumpArea_points], isClosed=True, color=(0, 0, 255), thickness=1)

        # 将绘制的多边形叠加在原图像上
        cv2.addWeighted(image, 0.5, image, 0.5, 0, image)

    def sort_points_clockwise(self, points):
        # 将列表转换为 NumPy 数组
        points = np.array(points)
        # 计算各点到参考点的极角
        reference_point = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - reference_point[1], points[:, 0] - reference_point[0])

        # 按极角排序
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        return sorted_points

    def extend_line(self, segment):
        # x1, y1 = segment[0][0], segment[0][1]  # 提取起点坐标
        # x2, y2 = segment[1][0], segment[1][1]  # 提取终点坐标
        #
        # # 计算线段的中点
        # mid_x = (x1 + x2) / 2
        # mid_y = (y1 + y2) / 2
        #
        # # 计算延长后的端点
        # extended_x1 = x1 + (x1 - mid_x)
        # extended_y1 = y1 + (y1 - mid_y)
        #
        # extended_x2 = x2 + (x2 - mid_x)
        # extended_y2 = y2 + (y2 - mid_y)
        #
        # return [[extended_x1, extended_y1], [extended_x2, extended_y2]]
        x1, y1 = segment[0][0], segment[0][1]  # 提取起点坐标
        x2, y2 = segment[1][0], segment[1][1]  # 提取终点坐标

        # 计算延长后的端点
        extended_x1 = x1 + 2 * (x1 - x2)
        extended_y1 = y1 + 2 * (y1 - y2)

        extended_x2 = x2 + 2 * (x2 - x1)
        extended_y2 = y2 + 2 * (y2 - y1)

        return [[extended_x1, extended_y1], [extended_x2, extended_y2]]

    def get_ruler_right_area(self):
        self.ruler_right_area = [self.real_ruler_lines[1][0], self.real_ruler_lines[1][1], self.real_ruler_lines[-1][0],
                                 self.real_ruler_lines[-1][1]]
