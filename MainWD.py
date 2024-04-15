# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWD.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1700, 906)
        MainWindow.setMinimumSize(QtCore.QSize(1700, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 120, 1280, 720))
        self.label.setStyleSheet("background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(255, 235, 235, 206), stop:0.35 rgba(255, 188, 188, 80), stop:0.4 rgba(255, 162, 162, 80), stop:0.425 rgba(255, 132, 132, 156), stop:0.44 rgba(252, 128, 128, 80), stop:1 rgba(255, 255, 255, 0));\n"
"border-radius:28px;")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(440, 30, 641, 81))
        self.label_2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.label_2.setStyleSheet("font:78px \"楷体\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(300, 10, 121, 121))
        self.label_3.setStyleSheet("")
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("UI/icon.png"))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, -10, 1701, 871))
        self.label_4.setStyleSheet("border-radius:35px;\n"
"background-color: rgb(167, 167, 167);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1340, 20, 101, 111))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("UI/avatar.png"))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1300, 10, 381, 201))
        self.label_6.setStyleSheet("background-color: rgb(101, 219, 255);\n"
"border-radius:8px;")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1320, 140, 131, 41))
        self.label_7.setStyleSheet("font:28px \"楷体\";")
        self.label_7.setScaledContents(False)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(1460, 30, 91, 31))
        self.label_8.setStyleSheet("font:23px \"楷体\";")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(1460, 150, 91, 31))
        self.label_9.setStyleSheet("font:23px \"楷体\";")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1460, 90, 91, 31))
        self.label_10.setStyleSheet("font:23px \"楷体\";")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(1560, 30, 91, 31))
        self.label_11.setStyleSheet("font:23px \"楷体\";")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(1560, 90, 91, 31))
        self.label_12.setStyleSheet("font:23px \"楷体\";")
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(1560, 150, 91, 31))
        self.label_13.setStyleSheet("font:23px \"楷体\";")
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1390, 220, 201, 51))
        self.pushButton.setStyleSheet("QPushButton{\n"
"font: 10pt \"宋体\";\n"
"border-radius:5px;\n"
"margin:1px;\n"
"color: black;\n"
"padding:2px 2px;\n"
"font:30px;\n"
"background-color: rgb(145,216,228);\n"
"    \n"
"}\n"
"QPushButton:hover{\n"
"background-color: rgb(145,216,228);\n"
"color:black;\n"
"\n"
"border:1px solid white;\n"
"}\n"
"QPushButton:pressed{\n"
"background-color: rgb(255, 255, 255);\n"
"color: black;\n"
"\n"
"border:1px solid white;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(1310, 290, 371, 111))
        self.label_14.setStyleSheet("border-radius:8px;font:43px \"楷体\";\n"
"background-color: rgb(235, 206, 100);")
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(1310, 290, 81, 111))
        self.label_15.setStyleSheet("border-radius:8px;font:30px \"楷体\";\n"
"background-color: rgb(255, 170, 0);")
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(1310, 420, 371, 361))
        self.label_16.setStyleSheet("border-radius:8px;font:43px \"楷体\";\n"
"background-color: rgb(235, 206, 100);")
        self.label_16.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_16.setObjectName("label_16")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1390, 790, 201, 51))
        self.pushButton_2.setStyleSheet("QPushButton{\n"
"font: 10pt \"宋体\";\n"
"border-radius:5px;\n"
"margin:1px;\n"
"color: black;\n"
"padding:2px 2px;\n"
"font:30px;\n"
"    background-color: rgb(229, 229, 0);\n"
"    \n"
"}\n"
"QPushButton:hover{background-color: rgb(229, 229, 0);\n"
"color:black;\n"
"\n"
"border:1px solid white;\n"
"}\n"
"QPushButton:pressed{\n"
"background-color: rgb(255, 255, 255);\n"
"color: black;\n"
"\n"
"border:1px solid white;\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(1320, 470, 351, 301))
        self.textEdit.setStyleSheet("font:43px \"楷体\";\n"
"background-color: rgb(235, 206, 100);")
        self.textEdit.setObjectName("textEdit")
        self.label_4.raise_()
        self.label_6.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_5.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.label_13.raise_()
        self.pushButton.raise_()
        self.label_14.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.pushButton_2.raise_()
        self.textEdit.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1700, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "跳远运动检测系统"))
        self.label_7.setText(_translate("MainWindow", "李华"))
        self.label_8.setText(_translate("MainWindow", "第一次："))
        self.label_9.setText(_translate("MainWindow", "第三次："))
        self.label_10.setText(_translate("MainWindow", "第二次："))
        self.pushButton.setText(_translate("MainWindow", "开始测试"))
        self.label_14.setText(_translate("MainWindow", "1/233"))
        self.label_15.setText(_translate("MainWindow", "测试\n"
"人数"))
        self.label_16.setText(_translate("MainWindow", "人员表"))
        self.pushButton_2.setText(_translate("MainWindow", "开始检录"))
        self.menu.setTitle(_translate("MainWindow", "跳远运动检测系统"))