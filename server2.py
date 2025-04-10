import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from src.predict import predict_coordinates  # Import the prediction function

class MyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            print("\n" + "="*40)
            print("Received prediction request:")
            
            # Process each attacker's data
            for attacker_data in data:
                attacker_id = attacker_data['attackerID']
                positions = attacker_data['positions']
                
                # Extract positions directly without validation
                # Format: [T-3.col, T-3.row, T-2.col, T-2.row, T-1.col, T-1.row, T.col, T.row]
                t2_col, t2_row = positions[2], positions[3]
                t1_col, t1_row = positions[4], positions[5]
                current_col, current_row = positions[6], positions[7]
                
                # Get predictions
                try:
                    pred = predict_coordinates(
                        p2x=t2_col, p2y=t2_row,
                        p1x=t1_col, p1y=t1_row,
                        cx=current_col, cy=current_row
                    )
                    print(f"\nAttacker {attacker_id} prediction:")
                    print(f"T-2 Position: ({t2_col}, {t2_row})")
                    print(f"T-1 Position: ({t1_col}, {t1_row})")
                    print(f"Current Position: ({current_col}, {current_row})")
                    print(f"Primary prediction: ({pred[0]}, {pred[1]}) {pred[2]*100:.1f}%")
                    print(f"Secondary prediction: ({pred[3]}, {pred[4]}) {pred[5]*100:.1f}%")
                except Exception as e:
                    print(f"Prediction failed for {attacker_id}: {str(e)}")
            
            print("="*40 + "\n")
                    
        except json.JSONDecodeError:
            print("Error decoding JSON payload")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "success"}).encode())

    def log_message(self, format, *args):
        return  # Disable request logging

def run(server_class=HTTPServer, handler_class=MyHandler, port=5001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Prediction server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
    run()