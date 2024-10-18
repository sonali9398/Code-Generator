from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
#from serpapi.google_search_results import GoogleSearch

#from serpapi import GoogleSearch
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Flask app
app = Flask(__name__)

# Load the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        return "".join([chunk.text for chunk in response])
    except Exception as e:
        return f"Error: {str(e)}"

def google_search(query):
    try:
        params = {
            "api_key": os.getenv("SERP_API_KEY"),
            "engine": "google",
            "q": query,
            "num": 1
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results and results["organic_results"]:
            return results["organic_results"][0]["link"]
        else:
            return "No results found."
    except Exception as e:
        return f"Error during Google Search: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    response = ""
    if "latest" in question.lower() or "current" in question.lower():
        response = google_search(question)
    else:
        response = get_gemini_response(question)
        if not response.strip() or "not available" in response.lower():
            response = google_search(question)
    return jsonify({"response": response})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"result": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "No selected file"}), 400

    if file:
        # Secure the filename and save the file (optional)
        filename = secure_filename(file.filename)
        file.save(os.path.join("uploads", filename))

        # Extract text from the file
        file.seek(0)  # Rewind the file pointer to the beginning
        text_content = file.read().decode('utf-8')  # Assuming the file is in UTF-8 format

        # Process the extracted text (e.g., send it to the Gemini model or use it directly)
        result = f"Extracted text from {file.filename}:\n{text_content}"
        return jsonify({"result": result})
    else:
        return jsonify({"result": "File upload failed"}), 400


if __name__ == '__main__':
    app.run(debug=True)
