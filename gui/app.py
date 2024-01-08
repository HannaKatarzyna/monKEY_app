import torch
import numpy as np
import sys
sys.path.append('../')
from kmodule.myMLP import *
from kmodule.keystroke_module import feature_extract_method_2
from main_window_ui import Ui_Dialog
from loginForm_ui import Ui_monKEY
from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QMainWindow, QMessageBox
)
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5 import QtCore
from PyQt5.QtMultimedia import QSound
from argon2.exceptions import VerifyMismatchError
from argon2 import PasswordHasher
import re
import os
import pandas as pd
from datetime import datetime


def count_time_from_0(df):
    s1 = pd.Series(df.iloc[1:]).reset_index(drop=True)
    s2 = pd.Series(df.iloc[:-1]).reset_index(drop=True)
    time_diff = (s1 - s2).apply(lambda x: x.total_seconds())
    time_diff = pd.concat([pd.Series(0.0), time_diff], ignore_index=True)
    return time_diff.cumsum()


def measure_ended(file):
    df = pd.read_csv(file, delimiter=" ", index_col=False,
                     header=None, names=['Press', 'Release', 'Hand'])
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

    va_HT = feature_extract_method_2(
        df, n_features=12, dynamic_feature='holdTime', time_feature='timeLapse', assumed_length=120, window_time=15)
    va_NFT = feature_extract_method_2(
        df, n_features=12, dynamic_feature='flightTime', time_feature='timeLapse', assumed_length=120, window_time=15, normalize_option=True)
    va = np.concatenate([va_HT[0], va_NFT[0]], axis=0)
    va = va.reshape((1, -1))

    # load model from pickle file (PyTORCH version)
    model = MLP.load_from_checkpoint(
        '../models/model_mlp.ckpt', map_location=torch.device("cpu"), encoder_map_location=torch.device("cpu"))
    model.eval()
    Y = model(torch.Tensor(va))
    Y = np.round(np.squeeze(Y.detach().numpy()))

    # # load model from pickle file (SKLEARN version)
    # with open('../models/model_svm.pkl', 'rb') as file:
    #     model = pickle.load(file)
    # Y = model.predict(va)
    return Y


def check_password_validity(password):
    while True:
        if (len(password) < 8):
            flag = -1
            break
        elif not re.search("[a-z]", password):
            flag = -1
            break
        elif not re.search("[A-Z]", password):
            flag = -1
            break
        elif not re.search("[0-9]", password):
            flag = -1
            break
        else:
            flag = 0
            print("Valid Password")
            break
    if flag == -1:
        print("Not a Valid Password ")


class Window(QWidget, Ui_Dialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.resultLabel.hide()
        self.lcdNumber.display("00:00")
        self.sound_file = QSound("podcast.wav")
        self.timestamp = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.timeUp)
        self.connectSignalsSlots()

    def timeUp(self):
        if self.timestamp != None:
            ts = datetime.now() - self.timestamp
            self.lcdNumber.display(str(ts)[2:7])

    def startRecording(self):
        print('The START button has been clicked')
        self.resultLabel.hide()
        self.textRecord.clear()
        self.textRecord.flag = True
        self.timer.start(1000)
        self.timestamp = datetime.now()
        self.sound_file.play()

    def stopRecording(self):
        print('The STOP button has been clicked')
        self.textRecord.flag = False
        self.timer.stop()
        self.timestamp = None
        self.sound_file.stop()
        self.lcdNumber.display("00:00")

        Y = measure_ended('exam_cor.txt')
        if Y:
            text = "Motor functions disorder was detect. Please, contact with your doctor."
        else:
            text = "Any motor functions disorder was detect."
        self.resultLabel.setText("Your result: "+text)
        self.resultLabel.show()
        if os.path.isfile("exam.txt"):
            os.remove("exam.txt")

    def connectSignalsSlots(self):
        self.startButton.clicked.connect(self.startRecording)
        self.stopButton.clicked.connect(self.stopRecording)


class initWindow(QMainWindow, Ui_monKEY):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.clearing_reg()
        self.win = None
        self.con = QSqlDatabase.addDatabase('QSQLITE')
        self.con.setDatabaseName("../database/DatabaseMonKEY.db")
        self.connectSignalsSlots()

    @staticmethod
    def info_box(text):
        msg = QMessageBox()
        msg.setWindowTitle('Message')
        msg.setStyleSheet(
            "QLabel{min-width: 400px; min-height: 100px; font-size: 15px;}")
        msg.setText(text)
        msg.exec_()

    def clearing_log(self):
        self.usernameEdit.clear()
        self.passwordEdit.clear()

    def clearing_reg(self):
        self.rep_passwordEdit.hide()
        self.rep_passwordLabel.hide()
        self.rep_passwordEdit.clear()
        self.loginButton.show()

    def loggingOut(self):
        self.win.close()
        self.clearing_log()
        self.show()

    def switchWindows(self):
        self.hide()
        self.win = Window()
        self.win.show()
        self.win.logoutButton1.clicked.connect(self.loggingOut)
        self.win.logoutButton2.clicked.connect(self.loggingOut)

    def user_login(self):
        out_username = self.usernameEdit.text()
        out_password = self.passwordEdit.text()
        ph = PasswordHasher()

        opened = self.con.open()
        if not opened:
            print("database not found!")
        else:
            query = QSqlQuery()
            query.exec(
                f"""SELECT password FROM users WHERE username=('{out_username}')""")
            query.first()
            if query.isNull(0):
                self.info_box("<p align='center'>User does not exist.")
            else:
                hashed_pass = query.value(0)
                try:
                    isValid = ph.verify(hashed_pass, str(out_password))
                    self.switchWindows()
                except VerifyMismatchError:
                    self.info_box("<p align='center'>Password is incorrect.")
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
            # query_str = f"""SELECT EXISTS(SELECT 1 FROM users WHERE username=('{out_username}'))"""
            query_existing.exec(
                f"""SELECT EXISTS(SELECT 1 FROM users WHERE username=('{out_username}'))""")
            query_existing.first()
            if query_existing.value(0):
                self.info_box(
                    "<p align='center'>This username exists in database.")
            else:
                if len(out_password) >= 8 and out_password == rep_password:

                    ph = PasswordHasher()
                    out_hashed = ph.hash(str(out_password))
                    query = QSqlQuery()
                    query.exec(
                        f"""INSERT INTO users (username, password) VALUES ('{out_username}','{out_hashed}')""")
                    self.con.close()
                    self.clearing_log()
                    self.clearing_reg()
                    self.info_box("<p align='center'>Successful registration.")

                else:
                    self.info_box(
                        "<p align='center'>Repeated password does not match.")

    def user_preregister(self):
        if self.loginButton.isVisible():

            self.loginButton.hide()
            self.rep_passwordEdit.show()
            self.rep_passwordLabel.show()
        else:
            self.user_register()

    def connectSignalsSlots(self):
        self.loginButton.clicked.connect(self.user_login)
        self.registerButton.clicked.connect(self.user_preregister)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    init_win = initWindow()
    init_win.show()
    sys.exit(app.exec())
