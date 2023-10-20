import sys
sys.path.append('../')
import pandas as pd
from datetime import datetime
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QMainWindow, QMessageBox
)

from kmodule.keystroke_module import feature_extract_method_1
from main_window_ui import Ui_Dialog
from loginForm_ui import Ui_MainWindow
from PyQt5.uic import loadUi


def count_time_from_0(df):
    s1 = pd.Series(df.iloc[1:]).reset_index(drop=True)
    s2 = pd.Series(df.iloc[:-1]).reset_index(drop=True)
    time_diff = (s1 - s2).apply(lambda x: x.total_seconds())
    time_diff = pd.concat([pd.Series(0.0), time_diff], ignore_index=True)
    return time_diff.cumsum()


class Window(QWidget, Ui_Dialog):

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

        df = pd.read_csv('exam.txt', delimiter=" ", index_col=False,
                         header=None, names=['Press', 'Release'])
        df_p = df['Press'].apply(lambda x: datetime.strptime(x, '%H:%M:%S.%f'))
        df_r = df['Release'].apply(
            lambda x: datetime.strptime(x, '%H:%M:%S.%f'))
        df['holdTime'] = (df_r - df_p).apply(lambda x: x.total_seconds())
        df['timeLapse'] = count_time_from_0(df_p)

        df_p = df_p.iloc[1:].reset_index(drop=True)
        df_r = df_r.iloc[:-1].reset_index(drop=True)
        df['flightTime'] = pd.concat([pd.Series(
            0.0), (df_p-df_r).apply(lambda x: x.total_seconds())], ignore_index=True)
        df['latencyTime'] = df['flightTime'] + \
            pd.concat([pd.Series(0.0), df['holdTime']], ignore_index=True)

        print(df.head())
        va_HT = feature_extract_method_1(
                    df, dynamic_feature='holdTime', time_feature='timeLapse', assumed_length=180, window_time=90)

        # load model from pickle file
        with open('model_2.pkl', 'rb') as file:  
            model = pickle.load(file)

        # evaluate model 
        Y = model.predict(va_HT.reshape((1, -1)))
        print("Result: ", Y)


    def connectSignalsSlots(self):
        self.startButton.clicked.connect(self.startRecording)
        self.stopButton.clicked.connect(self.stopRecording)
        # self.logoutButton.clicked.connect(self.close)

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

class initWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.win = None
        self.connectSignalsSlots()

    def switchWindows(self):
        print('The LOG IN button has been clicked')
        # first check password/user
        self.hide()
        self.win = Window()
        self.win.show()

    def connectSignalsSlots(self):
        self.loginButton.clicked.connect(self.switchWindows)
        # self.action_Exit.triggered.connect(self.close)
        # self.action_Find_Replace.triggered.connect(self.findAndReplace)
        # self.action_About.triggered.connect(self.about)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    init_win = initWindow()
    # win = Window()
    init_win.show()
    sys.exit(app.exec())
