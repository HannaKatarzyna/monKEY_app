# MasterDiploma - monKEY App
This application was designed as part of master thesis. The aim of monKEY is to demonstrate the algorithms' performence and effect. The software should be considered as prototype of the solution.

Create virtual environment:
```
python -m venv .venv
```
Activate your environment:
```
.venv\Scripts\activate.bat 
```
Manage packages:
```
pip freeze > requirements.txt 
or
pip install -r .\requirements.txt  
```
Run designer:
```
qt5-tools designer
```
Generate PyQt5 UI code:
```
pyuic5 -x loginForm.ui -o loginForm_ui.py
```
<!-- Create database in cmd:
```
sqlite3 DatabaseMonKEY.db
sqlite3 DatabaseMonKEY.db < insert_table.sql
```
In sqlite:
```
.tables
``` -->

<!-- Custom widgets:
```
from custom_widgets import MyQTextEdit
# self.textRecord = MyQTextEdit(Dialog)
# self.textRecord = MyQTextEdit(self.verticalLayoutWidget)
``` -->
