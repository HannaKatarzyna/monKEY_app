import sys
sys.path.append('../')
import pickle
from datetime import datetime
import pandas as pd
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QMainWindow, QMessageBox
)
from loginForm_ui import Ui_monKEY
from main_window_ui import Ui_Dialog
from kmodule.keystroke_module import feature_extract_method_1


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

class initWindow(QMainWindow, Ui_monKEY):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.rep_passwordEdit.hide()
        self.rep_passwordLabel.hide()
        self.win = None
        self.con = QSqlDatabase.addDatabase('QSQLITE')
        self.con.setDatabaseName("C:\sqlite\DatabaseMonKEY.db")
        self.connectSignalsSlots()

    # def resizeEvent(self, event):
    #     print("resizing is not able now")
    #     # QMainWindow.resizeEvent(self, event)

    @staticmethod
    def logging_failure():
        msg = QMessageBox()
        msg.setWindowTitle('Message')
        msg.setStyleSheet(
            "QLabel{min-width: 400px; min-height: 100px; font-size: 15px;}")
        msg.setText("<p align='center'>Incorrect Username or Password.")
        msg.exec_()

    def switchWindows(self):
        self.hide()
        self.win = Window()
        self.win.show()

    def first_check_password(self):
        out_username = self.usernameEdit.text()
        out_password = self.passwordEdit.text()
        ph = PasswordHasher()

        opened = self.con.open()
        if not opened:
            print("database not found!")
        else:
            query = QSqlQuery()
            query.exec(f"""SELECT password FROM users WHERE username=('{out_username}')""")
            query.first()
            if query.isNull(0):
                print('User does not exist.')
                self.logging_failure()
            else:
                hashed_pass = query.value(0)
                try:
                    isValid = ph.verify(hashed_pass, str(out_password))
                    self.switchWindows()
                except VerifyMismatchError:
                    print('Password is incorrect.')
                    self.logging_failure()
            self.con.close()      

    def user_register(self):

        out_username = self.usernameEdit.text()
        out_password = self.passwordEdit.text()
        rep_password = self.rep_passwordEdit.text()

        opened = self.con.open()
        if not opened:
            print("database not found!")
        else:
            query_existing = QSqlQuery()
            query_str = f"""SELECT EXISTS(SELECT 1 FROM users WHERE username=('{out_username}'))"""
            query_existing.exec(f"""SELECT EXISTS(SELECT 1 FROM users WHERE username=('{out_username}'))""")
            query_existing.first()
            if query_existing.value(0):
                print('This username exists in database.\nPlease select different name.')
                self.logging_failure()  
            else:
                if out_password == rep_password:

                    ph = PasswordHasher()
                    out_hashed = ph.hash(str(out_password))
                    query = QSqlQuery()
                    query.exec(f"""INSERT INTO users (username, password) VALUES ('{out_username}','{out_hashed}')""")
                    self.con.close()
                    print('Success.')
                    self.usernameEdit.clear()
                    self.passwordEdit.clear()
                    self.rep_passwordEdit.clear()
                    self.rep_passwordEdit.hide()
                    self.rep_passwordLabel.hide()
                    self.loginButton.show()
                        
                else:
                    print('Repeated password does not match.')
                    self.logging_failure()

    def user_preregister(self):
        if self.loginButton.isVisible():

            self.loginButton.hide()
            self.rep_passwordEdit.show()
            self.rep_passwordLabel.show()
        else:
            self.user_register()

    def connectSignalsSlots(self):
        self.loginButton.clicked.connect(self.first_check_password)
        self.registerButton.clicked.connect(self.user_preregister)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    init_win = initWindow()
    init_win.show()
    sys.exit(app.exec())


# TO DO: register FORM: change layouts, 
# TO DO: add limits for user chars number and password limits 
# TO DO: main WINDOW GUI: time 