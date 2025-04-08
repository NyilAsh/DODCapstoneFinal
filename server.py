import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class MyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # Respond to preflight requests with CORS headers.
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        # Read the content length and then the posted data
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            # Try to decode the JSON payload
            data = json.loads(post_data)
            print("Received data from client:", data)
        except json.JSONDecodeError:
            print("Error decoding JSON payload.")
        # Send a response back
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        response = {"status": "success"}
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        # Override to suppress logging of every request.
        return

def run(server_class=HTTPServer, handler_class=MyHandler, port=5001):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Extra processing server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
