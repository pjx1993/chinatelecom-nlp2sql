from flask import Flask, jsonify
from flask_cors import CORS
from flask import request,render_template

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
CORS(app)

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}

@app.route('/')
def api_root():
  return render_template('index.html')

@app.route('/parse_sql', methods=['POST'])
def parse_sql():

    task = {
        'table_id': request.json['table_id'],
        'question': request.json['question']
    }
    return jsonify({'task': request.json['question']})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=18201)
