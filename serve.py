from waitress import serve
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))
from app import app

if __name__ == '__main__':
    print("Starting Production Server for Parkinson's Detection System using Waitress...")
    print("Accessible locally at: http://localhost:5000")
    print("Accessible on your network via your local IP address over port 5000.")
    # Serve on all interfaces so mobile devices can connect
    serve(app, host='0.0.0.0', port=5000, threads=4)
