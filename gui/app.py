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

    def startRecording(self):
        print('The START button has been clicked')
        self.textRecord.flag = True

    def stopRecording(self):
        print('The STOP button has been clicked')
        self.textRecord.flag = False

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
