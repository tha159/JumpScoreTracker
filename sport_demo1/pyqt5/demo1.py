class Jumping(QObject):
    def __init__(self, ruler_seeker, name):
        super().__init__()
        self.imageProcessed = pyqtSignal(QImage)  # 定义为实例属性
        # 其他初始化代码...

    def startJump(self, remaining_chance):
        # 处理帧图片...
        self.draw_image()

    def draw_image(self):
        self.frameImage = cv2.cvtColor(self.frameImage, cv2.COLOR_RGB2BGR)
        h, w, c = self.frameImage.shape
        qImg = QImage(self.frameImage.data, w, h, w * c, QImage.Format_RGB888)
        self.imageProcessed.emit(qImg)  # 使用实例属性的信号

class loadMain(QMainWindow,MainWD.Ui_MainWindow,BaseClass):
    def __init__(self, parent=None):
        super().__init__(parent=None)
        self.setupUi(self)  # 加载ui
        self.center()

        print("窗口加载完毕！")

        # #无框和阴影
        self.setWindowFlag(Qt.FramelessWindowHint)  # 将界面设置为无框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 将界面属性设置为透明
        self.shadow = QGraphicsDropShadowEffect()  # 设定一个阴影,半径为10,颜色为#444444,定位为0,0
        self.shadow.setBlurRadius(10)
        self.shadow.setColor(QColor("#444444"))
        self.shadow.setOffset(0, 0)

        self.pushButton.clicked.connect(self.start_test)  # 开始测试
        self.pushButton_2.clicked.connect(self.start_check)  # 开始检录


    def start_test(self):
        test_images = 'img/2findRuler.jpg'
        ruler_template = 'img/ruler_template.jpg'
        video_name = 'img/jump_for_test_2.avi'
        names = ['张三', '李四', '王五', '李华']

        remaining_chance = 3
        ruler_seeker = RulerSeeker(test_images, ruler_template)
        jumping = Jumping(ruler_seeker, names[0])
        # 将Jumping类的imageProcessed信号连接到updateImage方法
        jumping.imageProcessed.connect(self.updateImage)
        jumping.startJump(remaining_chance)

    def updateImage(self, qImg):
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)