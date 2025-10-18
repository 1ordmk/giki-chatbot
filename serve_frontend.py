from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    os.chdir(frontend_dir)
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f'Serving frontend from {frontend_dir} at http://localhost:8000')
    httpd.serve_forever()