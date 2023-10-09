import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from main_window_ui import Ui_Dialog


class Window(QMainWindow, Ui_Dialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()
        self.flag = 0

    def keyPressEvent(self, event):
        pressed = event.key()
        print(pressed)
        # if (pressed in self.keys):
        #     index = self.keys.index(pressed)
        #     self.dots[index] = self.height()+self.upper
        #     self.repaint()
        if self.flag == 1:
            with open('exam_09_10.txt', 'a') as rec_file:
                rec_file.writelines()
        event.accept()

    def keyReleaseEvent(self, event):
        pressed = event.key()
        print(pressed)
        # if (pressed in self.keys):
        #     index = self.keys.index(pressed)
        #     self.dots[index] = self.lower
        #     self.repaint()
        if self.flag == 1:
            with open('reports_06_10.txt', 'a') as rec_file:
                rec_file.writelines()
        event.accept()

    def startRecording(self):
        print('The START button has been clicked')
        self.flag = 1

    def stopRecording(self):
        print('The STOP button has been clicked')
        self.flag = 0

# https://stackoverflow.com/questions/14159318/pyqt4-holding-down-a-key-detected-as-frequent-press-and-release
# https://stackoverflow.com/questions/49022442/pyqt-equivalent-of-keydown-event
    def connectSignalsSlots(self):
        self.startButton.clicked.connect(self.startRecording)
        self.stopButton.clicked.connect(self.stopRecording)
        # self.action_Exit.triggered.connect(self.close)
        # self.action_Find_Replace.triggered.connect(self.findAndReplace)
        # self.action_About.triggered.connect(self.about)

    def about(self):

        QMessageBox.about(
            self,
            "About Sample Editor",
            "<p>A sample text editor app built with:</p>"
            "<p>- PyQt</p>"
            "<p>- Qt Designer</p>"
            "<p>- Python</p>",
        )


# class FindReplaceDialog(QDialog):

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         loadUi("ui/find_replace.ui", self)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
