from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'Server is working!'})

if __name__ == '__main__':
    try:
        app.config['DEBUG'] = False
        app.run(host='127.0.0.1', port=5003)
    except Exception as e:
        print(f"Error starting server: {e}")