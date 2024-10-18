from flask import Flask, render_template, request, jsonify, send_file, session
import os
import google.generativeai as genai
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
import uuid
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found in environment variables.")
genai.configure(api_key=api_key)

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'xlsx', 'csv'}
app.secret_key = os.urandom(24)  # Set a secret key for session management

# Load the Gemini Pro model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(history=[])
except Exception as e:
    raise RuntimeError(f"Error initializing Google Gemini model: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file):
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    print(f"Extracting text from: {file_ext}")

    if file_ext == 'pdf':
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text

    elif file_ext in {'doc', 'docx'}:
        return docx2txt.process(file)

    elif file_ext == 'txt':
        return file.read().decode('utf-8')

    elif file_ext in {'png', 'jpg', 'jpeg'}:
        image = Image.open(file)
        return pytesseract.image_to_string(image)

    elif file_ext in {'xls', 'xlsx'}:
        df = pd.read_excel(file)
        return df.to_string(index=False)
    
    elif file_ext == 'csv':
        df = pd.read_csv(file)
        return df.to_string(index=False)

    else:
        return "Unsupported file type."

def get_gemini_response(question, textContent=""):
    try:
        prompt = f"Based on the following content:\n\n{textContent}\n\n{question}" if textContent else question
        response_text = ""

        for attempt in range(3):  # Retry up to 3 times
            try:
                response = chat.send_message(prompt, stream=True)
                for chunk in response:
                    if session.get('stop_signal', False):
                        break
                    if hasattr(chunk, 'text'):
                        response_text += chunk.text
                    else:
                        print("Chunk does not have 'text' attribute.")

                session['stop_signal'] = False
                return response_text or "No response received from the API."
            except Exception as e:
                if "429" in str(e):  # Check for rate limit error
                    print("Rate limit exceeded. Retrying...")
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    raise e
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

# @app.route('/ask', methods=['POST'])
# def ask():
#     question = request.form.get('question')
#     files = request.files.getlist('files')

#     print(f"Question: {question}")
#     print(f"Files: {files}")

#     if not question and not files:
#         return jsonify({"result": "Please provide a query or upload files."}), 400

#     if files:
#         responses_list = []
#         for file in files:
#             if file and allowed_file(file.filename):
#                 file.seek(0)
#                 textContent = extract_text_from_file(file)
#                 if textContent.strip():
#                     try:
#                         response = get_gemini_response(question, textContent)
#                         responses_list.append({
#                             "filename": file.filename,
#                             "response": response
#                         })
#                     except Exception as e:
#                         responses_list.append({
#                             "filename": file.filename,
#                             "response": f"Error processing file: {str(e)}"
#                         })
#                 else:
#                     responses_list.append({
#                         "filename": file.filename,
#                         "response": "No text extracted or text is empty."
#                     })
#             else:
#                 responses_list.append({
#                     "filename": file.filename,
#                     "response": "Invalid file type."
#                 })

#         return jsonify({"responses": responses_list})

#     return jsonify({"result": "No files found or invalid file type."}), 400

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    files = request.files.getlist('file')  # Adjusted to match input name

    if not question and not files:
        return jsonify({"result": "Please provide a query or upload files."}), 400

    responses_list = []
    for file in files:
        if file and allowed_file(file.filename):
            file.seek(0)
            textContent = extract_text_from_file(file)
            if textContent.strip():
                try:
                    response = get_gemini_response(question, textContent)
                    responses_list.append({
                        "filename": file.filename,
                        "response": response
                    })
                except Exception as e:
                    responses_list.append({
                        "filename": file.filename,
                        "response": f"Error processing file: {str(e)}"
                    })
            else:
                responses_list.append({
                    "filename": file.filename,
                    "response": "No text extracted or text is empty."
                })
        else:
            responses_list.append({
                "filename": file.filename,
                "response": "Invalid file type."
            })

    return jsonify({"responses": responses_list})


@app.route('/ask_voice', methods=['POST'])
def ask_voice():
    query = recognize_speech_from_mic()
    if query in ["API unavailable", "Unable to recognize speech"]:
        return jsonify({"result": query}), 400

    response = get_gemini_response(query)
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
            response = get_gemini_response(textContent)
            return jsonify({"result": response})
        else:
            return jsonify({"result": "No text extracted or text is empty."}), 400
    else:
        return jsonify({"result": "Invalid file type."}), 400

@app.route('/stop', methods=['POST'])
def stop_response():
    session['stop_signal'] = True
    return jsonify({"result": "Response generation stopped."}), 200

@app.route('/save_response', methods=['POST'])
def save_response():
    data = request.json
    response_text = data.get('response', '')
    response_id = str(uuid.uuid4())
    responses[response_id] = response_text
    return jsonify({'link': f'/view_response/{response_id}', 'qr_code': f'/generate_qr/{response_id}'}), 200

@app.route('/view_response/<response_id>')
def view_response(response_id):
    response_text = responses.get(response_id, 'Response not found')
    return f"<h1>Response:</h1><p>{response_text}</p>"

@app.route('/generate_qr/<response_id>')
def generate_qr(response_id):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(f'/view_response/{response_id}')
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    
    img_bytes = BytesIO()
    img.save(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png', as_attachment=True, download_name='qr_code.png')

if __name__ == '__main__':
    app.run(debug=True)
