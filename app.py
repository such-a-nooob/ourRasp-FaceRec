from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
import os
import config

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Function to connect to the found database
def get_found_db_connection():
    return sqlite3.connect(config.FOUND_DB_PATH)

# Function to connect to the user database
def get_user_db_connection():
    conn = sqlite3.connect(config.USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Create the user database and table if it doesn't exist
def initialize_user_db():
    conn = get_user_db_connection()
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dob TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

initialize_user_db()

# Home route (requires login)
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    current_date = datetime.now().strftime('%Y-%m-%d')
    conn = get_found_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, time, image_path FROM {config.TABLE_NAME} WHERE date = ?", (current_date,))
    found_data = cursor.fetchall()
    conn.close()

    return render_template('index.html', selected_date=current_date, found_data=found_data, username=session['username'])

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_user_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['username'] = f"{user['first_name']} {user['last_name']}"
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        dob = request.form['dob']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_user_db_connection()
        try:
            conn.execute('INSERT INTO users (first_name, last_name, dob, email, password) VALUES (?, ?, ?, ?, ?)',
                         (first_name, last_name, dob, email, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error='Email already exists')
    return render_template('register.html')

# Found route (requires login)
@app.route('/found', methods=['POST'])
def found():
    if 'username' not in session:
        return redirect(url_for('login'))

    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = get_found_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT name, time, image_path FROM {config.TABLE_NAME} WHERE date = ?", (formatted_date,))
    found_data = cursor.fetchall()
    conn.close()

    if not found_data:
        return render_template('index.html', selected_date=selected_date, no_data=True, username=session['username'])

    return render_template('index.html', selected_date=selected_date, found_data=found_data, username=session['username'])

# About route (requires login)
@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', username=session['username'])

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
