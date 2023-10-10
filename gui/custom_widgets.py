from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit
from datetime import datetime

class MyQTextEdit(QtWidgets.QTextEdit):

    flag: bool

    def __init__(self, UI):
        super().__init__(UI)
        self.flag = False

    def keyPressEvent(self, eventQKeyEvent): 
        super().keyPressEvent(eventQKeyEvent)
        ts = datetime.now().strftime('%H:%M:%S.%f')
        # pressed = eventQKeyEvent.key()
        # print('pressed:  ', pressed)
        if self.flag:
            with open('exam_press.txt', 'a') as p_file:
                p_file.writelines(str(ts)+'\n')

    def keyReleaseEvent(self, eventQKeyEvent):
        super().keyReleaseEvent(eventQKeyEvent)
        ts = datetime.now().strftime('%H:%M:%S.%f')
        # released = eventQKeyEvent.key()
        # print('released:  ', released)
        if self.flag and not eventQKeyEvent.isAutoRepeat():
            with open('exam_release.txt', 'a') as r_file:
                r_file.writelines(str(ts)+'\n')