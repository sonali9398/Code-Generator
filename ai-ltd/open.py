from http.client import responses
import time
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, session, redirect, url_for
from flask_session import Session
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx2txt
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import pandas as pd
import speech_recognition as sr
import qrcode
from io import BytesIO
from html import escape
from flask_cors import CORS
import openai

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

app.secret_key = '_5#y2L"F4Q8z\n\xec]/'
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 300
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
PERMANENT_SESSION_LIFETIME = 1800
app.config.update(SECRET_KEY=os.urandom(24))
app.config.from_object(__name__)
Session(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'xls', 'xlsx', 'csv'}

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("No OPENAI_API_KEY set for Flask application.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_file(file):
    file_ext = file.filename.rsplit('.', 1)[1].lower()

    print(f"Extracting text from: {file_ext}")

    if file_ext == 'pdf':
        reader = PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
            print(f"Extracted text: {text[:100]}")
        return text

    elif file_ext in {'doc', 'docx'}:
        return docx2txt.process(file)

    elif file_ext == 'txt':
        return file.read().decode('utf-8')

    elif file_ext in {'png', 'jpg', 'jpeg'}:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text

    elif file_ext in {'xls', 'xlsx'}:
        df = pd.read_excel(file)
        text = df.to_string(index=False)
        return text

    elif file_ext == 'csv':
        df = pd.read_csv(file)
        text = df.to_string(index=False)
        return text

    else:
        return "Unsupported file type."


def get_openai_response(question, textContent=""):
    try:
        prompt = f"Based on the following content:\n\n{textContent}\n\n{question}" if textContent else question
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",  # Specify the model engine
            prompt=prompt,
            max_tokens=150,  # Adjust as needed
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f"Recognized query: {query}")
        return query
    except sr.RequestError:
        return "API unavailable"
    except sr.UnknownValueError:
        return "Unable to recognize speech"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    files = request.files.getlist('file')

    if not question and not files:
        return jsonify({"result": "Please provide a query or upload a file."}), 400

    combined_text_content = ""

    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"result": f"Invalid file type for {file.filename}."}), 400

        file.seek(0)
        textContent = extract_text_from_file(file)
        if not textContent.strip():
            return jsonify({"result": f"No text extracted or text is empty for {file.filename}."}), 400

        combined_text_content += f"\n{textContent}"

    # Ensure that all text content is passed together with the question
    response = get_openai_response(question, combined_text_content)
    return jsonify({"response": response})


@app.route('/stop', methods=['POST'])
def stop_response():
    session['stop_signal'] = True
    return jsonify({"result": "Response generation stopped."}), 200


@app.route('/ask_voice', methods=['POST'])
def ask_voice():
    query = recognize_speech_from_mic()
    if query in ["API unavailable", "Unable to recognize speech"]:
        return jsonify({"result": query}), 400

    response = get_openai_response(query)
    return jsonify({"response": response})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        file.seek(0)  # Rewind the file pointer to the beginning
        textContent = extract_text_from_file(file)

        if textContent.strip():
            response = get_openai_response(textContent)
            return jsonify({"result": response})
        else:
            return jsonify({"result": "No text extracted or text is empty."}), 400
    else:
        return jsonify({"result": "Invalid file type."}), 400


responses = {}


@app.route('/save_response', methods=['POST'])
def save_response():
    data = request.json
    response_text = data.get('response', '')
    response_id = str(uuid.uuid4())
    file_path = f'responses/{response_id}.txt'

    # Create the responses directory if it doesn't exist
    if not os.path.exists('responses'):
        os.makedirs('responses')

    # Save the response text to a .txt file
    with open(file_path, 'w') as file:
        file.write(response_text)

    # Construct full URLs
    link = url_for('view_response', response_id=response_id, _external=True)
    qr_code = url_for('generate_qr', response_id=response_id, _external=True)

    return jsonify({'link': link, 'qr_code': qr_code}), 200


@app.route('/view_response/<response_id>')
def view_response(response_id):
    file_path = f'responses/{response_id}.txt'

    if not os.path.exists(file_path):
        return f"<h1>Response Not Found</h1>", 404

    with open(file_path, 'r') as file:
        response_text = file.read()

    return f"<h1>Response:</h1><p>{response_text}</p>"


@app.route('/generate_qr/<response_id>')
def generate_qr(response_id):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url_for('view_response', response_id=response_id, _external=True))
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')

    img_bytes = BytesIO()
    img.save(img_bytes)
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png', as_attachment=True, download_name='qr_code.png')


if __name__ == '__main__':
    with app.test_request_context("/"):
        session["key"] = "value"
    app.run(debug=True)
    CORS(app)
