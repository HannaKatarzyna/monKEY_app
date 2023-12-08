from PyQt5 import QtCore
from PyQt5.QtWidgets import QTextEdit
from datetime import datetime


def which_key(key):
    # patternL = 'q|w|e|r|t|a|s|d|f|g|z|x|c|v|b'
    patternL = [81, 87, 69, 82, 84, 65, 83, 68, 70, 71, 90, 88, 67, 86, 66]
    # patternR = 'y|u|i|o|p|h|j|k|l|n|m|comma|period|semicolon|slash'
    patternR = [89, 85, 73, 79, 80, 72, 74, 75, 76, 78, 77, 44, 46, 47, 59]
    if key == QtCore.Qt.Key_Space:
        return 'S'
    elif key in patternL:
        return 'L'
    elif key in patternR:
        return 'R'
    else:
        return 'N'


class MyQTextEdit(QTextEdit):

    flag: bool
    current_keys = {}

    # def __init__(self, UI, layout):
    #     super().__init__(UI, layout)
    def __init__(self, layout):
        super().__init__(layout)
        self.flag = False

    def keyPressEvent(self, eventQKeyEvent):
        super().keyPressEvent(eventQKeyEvent)
        if self.flag:
            ts = datetime.now().strftime('%H:%M:%S.%f')
            pressed = eventQKeyEvent.key()
            print(pressed)
            self.current_keys[pressed] = ts

    def keyReleaseEvent(self, eventQKeyEvent):
        super().keyReleaseEvent(eventQKeyEvent)
        if self.flag:
            ts = datetime.now().strftime('%H:%M:%S.%f')
            released = eventQKeyEvent.key()
            if released in self.current_keys and not eventQKeyEvent.isAutoRepeat():
                part_line = ' '+ts+' '+which_key(released)+'\n'
                with open('exam.txt', 'a') as file:
                    file.writelines(self.current_keys[released] + part_line)
                del self.current_keys[released]
