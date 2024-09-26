import sqlite3


conn = sqlite3.connect('history.db')
c = conn.cursor()

# Create the history table
c.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY,
        image_name TEXT UNIQUE,
        captions TEXT
    )
''')

conn.commit()
conn.close()