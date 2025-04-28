from flask import Flask, request
import os
from config import ALL_FACES_DIR

app = Flask(__name__)
os.makedirs(ALL_FACES_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if file:
        filename = file.filename
        try:
            parts = filename.split('_')
            if len(parts) < 2:
                raise ValueError("Invalid filename format")
            date_raw = parts[1]  # e.g., 20250428
            if len(date_raw) != 8 or not date_raw.isdigit():
                raise ValueError("Invalid date format in filename")
            # Format the date as YYYY-MM-DD
            formatted_date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
            
            save_dir = os.path.join(ALL_FACES_DIR, formatted_date)
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, filename)
            file.save(file_path)
            print(f"Image saved: {file_path}")
            return "Success", 200
        except Exception as e:
            print("Error saving file:", e)
            return "Invalid filename format", 400
    return "Failed", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
