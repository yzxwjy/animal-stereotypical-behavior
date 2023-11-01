import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, qRgb
from track import Ui_MainWindow
from PyQt5.QtWidgets import *
import sys

from subprocess import call
from auto import r

class Mainwindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        self.setupUi(self)
        self.controller()

    def controller(self):
        self.pushButton.clicked.connect(self.genzong)
        self.pushButton_cross.clicked.connect(self.cross_corr)
        self.pushButton_auto.clicked.connect(self.auto_corr)

    def genzong(self):
        call(['python', 'D:/yzx/siamban-master/tools/demo.py'])

    def cross_corr(self):
        call(['python', 'D:/yzx/siamban-master/cross.py'])
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.label.setPixmap(QPixmap(fname[0]))
            self.label.setWordWrap(True)
            self.label.setScaledContents(True)

    def auto_corr(self):
        call(['python', 'D:/yzx/siamban-master/auto.py'])
        #self.lineEdit.setText(str(r))
        if 0.015< r <0.20:
            QMessageBox.information(self, 'Classification of trajectory', 'circular curve')





if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Mainwindow()
    ui.show()
    sys.exit(app.exec_())
