# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(759, 620)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(15)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab1)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 80, 721, 351))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        # self.textRecord = QtWidgets.QTextEdit(self.verticalLayoutWidget)
        self.textRecord = MyQTextEdit(self.verticalLayoutWidget)
        self.textRecord.setStyleSheet("background-color: rgb(226, 234, 255);")
        self.textRecord.setObjectName("textRecord")
        self.verticalLayout.addWidget(self.textRecord)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab1)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(230, 430, 322, 81))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.startButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        self.stopButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout.addWidget(self.stopButton)
        self.formLayoutWidget = QtWidgets.QWidget(self.tab1)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 20, 141, 61))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.lcdNumber = QtWidgets.QLCDNumber(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber.sizePolicy().hasHeightForWidth())
        self.lcdNumber.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lcdNumber.setFont(font)
        self.lcdNumber.setAccessibleName("")
        self.lcdNumber.setStyleSheet("font: 14pt \"Rockwell\";")
        self.lcdNumber.setSmallDecimalPoint(False)
        self.lcdNumber.setObjectName("lcdNumber")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lcdNumber)
        self.logoutButton1 = QtWidgets.QCommandLinkButton(self.tab1)
        self.logoutButton1.setGeometry(QtCore.QRect(600, 0, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(15)
        self.logoutButton1.setFont(font)
        self.logoutButton1.setObjectName("logoutButton1")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab1)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 510, 721, 32))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.resultLabel = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.resultLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.resultLabel.setFont(font)
        self.resultLabel.setObjectName("resultLabel")
        self.horizontalLayout_2.addWidget(self.resultLabel)
        self.tabWidget.addTab(self.tab1, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.logoutButton2 = QtWidgets.QCommandLinkButton(self.tab2)
        self.logoutButton2.setGeometry(QtCore.QRect(600, 0, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(15)
        self.logoutButton2.setFont(font)
        self.logoutButton2.setObjectName("logoutButton2")
        self.tabWidget.addTab(self.tab2, "")
        self.verticalLayout_2.addWidget(self.tabWidget)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.textRecord.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Rockwell\'; font-size:15pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Click START button and write a text here...  After you finish, click STOP and wait a moment for results.</p></body></html>"))
        self.startButton.setText(_translate("Dialog", "START"))
        self.stopButton.setText(_translate("Dialog", "STOP"))
        self.logoutButton1.setText(_translate("Dialog", "LOG OUT"))
        self.resultLabel.setText(_translate("Dialog", "Your result:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("Dialog", "Test"))
        self.logoutButton2.setText(_translate("Dialog", "LOG OUT"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), _translate("Dialog", "About app"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
