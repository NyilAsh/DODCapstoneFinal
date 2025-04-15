from flask import Flask, request
from flask_cors import CORS
import csv
import os

app = Flask(__name__)
CORS(app)

CSV_FILE = 'data.csv'

def init_csv():
    # Create an empty CSV file if it doesn't exist yet.
    if not os.path.exists(CSV_FILE):
        open(CSV_FILE, 'w').close()

@app.route('/log', methods=['POST'])
def log_data():
    # Route for logging game data via POST requests.
    try:
        data = request.get_json()
        # Ensure the incoming data is in list format.
        if not isinstance(data, list):
            return {'status': 'error', 'message': 'Invalid data format'}, 400
            
        # Append each entry in the received list to our CSV file.
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for entry in data:
                # Expecting each entry to be a list of length 8.
                if isinstance(entry, list) and len(entry) == 8:
                    writer.writerow(entry)
        return {'status': 'success'}, 200
    except Exception as e:
        # In case of errors, print them and return a 500 status.
        print(f"Error logging data: {str(e)}")
        return {'status': 'error'}, 500

@app.route('/log', methods=['GET'])
def health_check():
    # Simple GET route that confirms the logger is running.
    return {'status': 'running'}, 200

if __name__ == '__main__':
    init_csv()
    # Start the Flask app on port 5000.
    print(f"Game logger running. CSV file: {CSV_FILE}")
    app.run(port=5000)
