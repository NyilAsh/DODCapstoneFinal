from flask import Flask, request
from flask_cors import CORS
import csv
import os

app = Flask(__name__)
CORS(app)

CSV_FILE = 'data.csv'

def init_csv():
    if not os.path.exists(CSV_FILE):
        open(CSV_FILE, 'w').close()

@app.route('/log', methods=['POST'])
def log_data():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return {'status': 'error', 'message': 'Invalid data format'}, 400
            
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for entry in data:
                if isinstance(entry, list) and len(entry) == 8:
                    writer.writerow(entry)
        return {'status': 'success'}, 200
    except Exception as e:
        print(f"Error logging data: {str(e)}")
        return {'status': 'error'}, 500

@app.route('/log', methods=['GET'])
def health_check():
    return {'status': 'running'}, 200

if __name__ == '__main__':
    init_csv()
    print(f"Game logger running. CSV file: {CSV_FILE}")
    app.run(port=5000)