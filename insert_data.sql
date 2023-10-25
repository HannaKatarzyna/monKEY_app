
DROP TABLE IF EXISTS users;

-- instrukcje do tworzenia odpowiednich tabel

CREATE TABLE users
(
	id INTEGER PRIMARY KEY AUTOINCREMENT, 
	username VARCHAR(50) NOT NULL,
	password CHAR(100) NOT NULL
);
