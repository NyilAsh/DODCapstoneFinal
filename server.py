import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from src.predict import predict_coordinates  # Import the prediction function

class MyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # Handle preflight (CORS) requests with the necessary headers.
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        # Handle incoming POST requests containing prediction data.
        content_length = int(self.headers.get('Content-Length', 0))
        # Read the exact number of bytes specified in the headers.
        post_data = self.rfile.read(content_length)
        try:
            # Convert the JSON payload into a Python object.
            data = json.loads(post_data)
            print("\n" + "="*40)
            print("Received prediction request:")
            
            predictions = []
            # Process each attacker's data in the request.
            for attacker_data in data:
                attacker_id = attacker_data['attackerID']
                positions = attacker_data['positions']
                
                # Unpack the positions from the array.
                t3x, t3y, t2x, t2y, t1x, t1y, cx, cy = positions
                
                # Call the ML/logic function to get predictions.
                pred1x, pred1y, pred1conf, pred2x, pred2y, pred2conf = predict_coordinates(
                    t2x, t2y,  # T-2 position
                    t1x, t1y,  # T-1 position
                    cx, cy     # Current position
                )
                
                # Print debug info to the console.
                print(f"\nAttacker {attacker_id} prediction:")
                print(f"T-2 Position: ({t2x}, {t2y})")
                print(f"T-1 Position: ({t1x}, {t1y})")
                print(f"Current Position: ({cx}, {cy})")
                print(f"Primary prediction: ({pred1x}, {pred1y}) {pred1conf*100:.1f}%")
                print(f"Secondary prediction: ({pred2x}, {pred2y}) {pred2conf*100:.1f}%")
                
                # Append the predictions to the response list.
                predictions.append({
                    'attackerID': attacker_id,
                    'predictions': [pred1x, pred1y, pred1conf, pred2x, pred2y, pred2conf]
                })
            
            print("="*40 + "\n")
            
            # Send back the prediction results as JSON.
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(predictions).encode())
            
        except Exception as e:
            # In case of exceptions, return a 500 status and log the error.
            print(f"Error processing request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        # Override the default logging to disable console logs for each request.
        return

def run(server_class=HTTPServer, handler_class=MyHandler, port=5001):
    # Set up the HTTP server.
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting prediction server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
