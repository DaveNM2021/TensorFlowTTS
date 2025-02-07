from flask import Flask, request, render_template, jsonify 
from tts import get_wav

app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello, World!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speak', methods=['POST'])
def speak():
    text = request.form['text']
    out_filename = get_wav(text)
    return jsonify({"audio_file": "static/" + out_filename})
                     
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8080)