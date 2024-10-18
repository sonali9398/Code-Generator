from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx2txt
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file):
    file_ext = file.filename.rsplit('.', 1)[1].lower()

    if file_ext == 'pdf':
        reader = PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text

    elif file_ext in {'doc', 'docx'}:
        return docx2txt.process(file)

    elif file_ext == 'txt':
        return file.read().decode('utf-8')

    elif file_ext in {'png', 'jpg', 'jpeg'}:
        image = Image.open(file)
        return pytesseract.image_to_string(image)

    else:
        return "Unsupported file type."

def get_gemini_response(question, text_content):
    try:
        # Combine user input question with extracted text
        prompt = f"Based on the following content:\n\n{text_content}\n\n{question}"
        response = chat.send_message(prompt, stream=True)
        return "".join([chunk.text for chunk in response])
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    file = request.files.get('file')
    
    if not file or not allowed_file(file.filename):
        return jsonify({"result": "Invalid file or no file uploaded"}), 400
    
    file.seek(0)
    text_content = extract_text_from_file(file)
    
    if text_content.strip():
        response = get_gemini_response(question, text_content)
        return jsonify({"response": response})
    else:
        return jsonify({"result": "No text extracted or text is empty."}), 400

if __name__ == '__main__':
    app.run(debug=True)
