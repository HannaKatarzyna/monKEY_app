from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit
from datetime import datetime

class MyQTextEdit(QtWidgets.QTextEdit):

    flag: bool
    current_keys: dict

    def __init__(self, UI):
        super().__init__(UI)
        self.flag = False
        self.current_keys = {}

    def keyPressEvent(self, eventQKeyEvent): 
        super().keyPressEvent(eventQKeyEvent)
        if self.flag:
            ts = datetime.now().strftime('%H:%M:%S.%f')
            pressed = eventQKeyEvent.key()
            self.current_keys[pressed] = ts
            print(self.current_keys)

    def keyReleaseEvent(self, eventQKeyEvent):
        super().keyReleaseEvent(eventQKeyEvent)
        if self.flag:
            ts = datetime.now().strftime('%H:%M:%S.%f')
            released = eventQKeyEvent.key()
            if released in self.current_keys and not eventQKeyEvent.isAutoRepeat():
                with open('exam.txt', 'a') as file:
                    file.writelines(self.current_keys[released]+' '+ts+'\n')
                del self.current_keys[released]
                print(self.current_keys)