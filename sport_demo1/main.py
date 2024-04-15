from RulerSeeker import RulerSeeker
from Jumping import Jumping
from UI import loadMain

import sys
from PyQt5.QtWidgets import QApplication,QWidget



##环境：相机和尺子保持不动。


def show_ruler():
    test_images = 'img/2findRuler.jpg'
    ruler_template = 'img/ruler_template.jpg'
    ruler_seeker = RulerSeeker(test_images, ruler_template)
    ruler_seeker.showInfo()


def show_jump():
    test_images = 'img/2findRuler.jpg'
    ruler_template = 'img/ruler_template.jpg'
    video_name = 'img/jump_for_test_2.avi'

    remaining_chance = 3
    ruler_seeker = RulerSeeker(test_images, ruler_template)
    jumping = Jumping(ruler_seeker, '李华')
    jumping.startJumpxxx(remaining_chance)


if __name__ == '__main__':
    # show_ruler()
    # show_jump()


    app = QApplication(sys.argv)
    loadWin = loadMain()
    loadWin.show()
    sys.exit(app.exec_())
