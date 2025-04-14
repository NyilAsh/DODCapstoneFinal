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
            
            predictions = []
            # Process each attacker's data
            for attacker_data in data:
                attacker_id = attacker_data['attackerID']
                positions = attacker_data['positions']
                
                # Extract positions from the array
                t3x, t3y, t2x, t2y, t1x, t1y, cx, cy = positions
                
                # Get predictions
                pred1x, pred1y, pred1conf, pred2x, pred2y, pred2conf = predict_coordinates(
                    t2x, t2y,  # T-2 position
                    t1x, t1y,  # T-1 position
                    cx, cy     # Current position
                )
                
                # Print predictions to terminal
                print(f"\nAttacker {attacker_id} prediction:")
                print(f"T-2 Position: ({t2x}, {t2y})")
                print(f"T-1 Position: ({t1x}, {t1y})")
                print(f"Current Position: ({cx}, {cy})")
                print(f"Primary prediction: ({pred1x}, {pred1y}) {pred1conf*100:.1f}%")
                print(f"Secondary prediction: ({pred2x}, {pred2y}) {pred2conf*100:.1f}%")
                
                # Add predictions to response
                predictions.append({
                    'attackerID': attacker_id,
                    'predictions': [pred1x, pred1y, pred1conf, pred2x, pred2y, pred2conf]
                })
            
            print("="*40 + "\n")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(predictions).encode())
            
        except Exception as e:
            print(f"Error processing request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        return  # Disable request logging

def run(server_class=HTTPServer, handler_class=MyHandler, port=5001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting prediction server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()