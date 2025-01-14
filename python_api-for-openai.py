from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import openai
from openai import OpenAI
from openai import AzureOpenAI
import os
import sys
import re
import hmac
import hashlib
import base64
import json
import time
import datetime
import logging
from dotenv import load_dotenv
import fitz
import importlib, json
from PIL import Image
import pytesseract
import io
import requests
import tempfile

# Create Flask app instance
app = Flask(__name__)
CORS(app, resources={
    r"/gpt/*": {
        "origins": ["http://localhost:3000", "https://app.associateattorney.ai"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

load_dotenv()  # This loads the variables from .env

OPENAI_AZURE_API_VERSION = os.getenv('OPENAI_AZURE_API_VERSION')
OPENAI_AZURE_API_BASE_URL = os.getenv('OPENAI_AZURE_API_BASE_URL')
OPENAI_AZURE_API_KEY = os.getenv('OPENAI_AZURE_API_KEY')
OPENAI_AZURE_API_ENGINE = os.getenv('OPENAI_AZURE_API_ENGINE')

def get_response_from_ai_gpt_4_32k(messages):
    client = AzureOpenAI(
        api_key=OPENAI_AZURE_API_KEY,
        api_version=OPENAI_AZURE_API_VERSION,
        azure_endpoint=OPENAI_AZURE_API_BASE_URL
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_AZURE_API_ENGINE,
            messages=messages
        )
        # Extract the response content
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def extract_file_content(url, file_type):
    # Add scheme if missing
    if url.startswith('/gitea'):
        url = f"http://localhost{url}"

    response = requests.get(url)
    if response.status_code != 200:
        return "[Error: Failed to fetch file]"

    if file_type.startswith('image/'):
        try:
            # First try using Tesseract if available
            try:
                image = Image.open(io.BytesIO(response.content))
                text = pytesseract.image_to_string(image)
                return f"[Image Content (OCR):\n{text}]"
            except (ImportError, EnvironmentError):
                # If Tesseract is not available, return image dimensions and basic info
                image = Image.open(io.BytesIO(response.content))
                width, height = image.size
                mode = image.mode
                format = image.format
                return f"[Image Info: {format} image, {width}x{height} pixels, {mode} mode]"
        except Exception as e:
            return f"[Error processing image: {str(e)}]"
            
    elif file_type == 'application/pdf':
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                
                pdf_document = fitz.open(temp_file.name)
                text = ""
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text += page.get_text()
                pdf_document.close()
                
                return f"[PDF Content:\n{text}]"
        except Exception as e:
            return f"[Error processing PDF: {str(e)}]"
            
    return "[Unsupported file type]"

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Add this before each route
@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.route('/gpt/get_ai_response', methods=['POST'])
def get_ai_response():
    prompt = request.json.get('prompt')
    system_prompt = request.json.get('systemPrompt')
    taskId = request.json.get('taskId')
    matterId = request.json.get('matterId')

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = get_response_from_ai_gpt_4_32k(messages)
    return jsonify({"response": response, "systemPrompt": system_prompt, "userPrompt": prompt})

@app.route('/gpt/extract_file_content', methods=['POST'])
def get_file_content():
    url = request.json.get('url')
    file_type = request.json.get('fileType')
    
    try:
        content = extract_file_content(url, file_type)
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"content": f"[Error: {str(e)}]"})

@app.route('/gpt/get_typeahead_suggestions', methods=['POST'])
def get_typeahead_suggestions():
    current_text = request.json.get('text')
    cursor_position = request.json.get('cursorPosition')
    system_prompt = request.json.get('systemPrompt')
    
    # Get the complete text before cursor for context
    text_before_cursor = current_text[:cursor_position]
    
    # Find the word at cursor (complete or incomplete)
    start = text_before_cursor.rstrip().rfind(' ') + 1 if ' ' in text_before_cursor.rstrip() else 0
    current_word = text_before_cursor[start:].strip()
    
    # Get the last few sentences for context (up to 500 characters)
    context_text = text_before_cursor[-500:] if len(text_before_cursor) > 500 else text_before_cursor
    
    if not current_word and not context_text:  # Only suggest if there's some input
        return jsonify({"suggestions": []})
        
    messages = [
        {"role": "system", "content": f"""You are a specialized legal assistant for a California civil law practice. Your role is to provide a single, detailed paragraph completion or continuation based on the following context:

1. Family Law: Divorce proceedings, child custody, spousal support, visitation rights, domestic partnerships, prenuptial agreements, child support calculations

2. Housing Law: Landlord-tenant disputes, eviction proceedings, rental agreements, lease violations, property maintenance, fair housing

3. Contract Law: Contract drafting/review, breach of contract, service agreements, business contracts, settlements

4. Employment Law: Workplace discrimination, wrongful termination, wage disputes, employment contracts, harassment, workers' compensation

5. Property Law: Real estate transactions, boundary disputes, easements, HOA disputes, title issues

6. Civil Harassment: Restraining orders, civil harassment orders, workplace protection, elder abuse

Consider:
- California legal terminology
- Civil Code sections
- Standard legal document phrases
- Case citation formats
- Civil litigation terms
- California court forms
- Legal pleading language

Previous context: {context_text}
Current word: {current_word}

Based on the previous context and current word, provide ONE detailed paragraph suggestion that would naturally continue the legal document. Format as a single string without any special formatting or markers."""},
        {"role": "user", "content": f"Previous text: {context_text}\nCurrent word: {current_word}"}
    ]
    
    try:
        response = get_response_from_ai_gpt_4_32k(messages)
        suggestions = [response.strip()]
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/gpt/process_consultation', methods=['POST'])
def process_consultation():
    try:
        # Get request data
        data = request.json
        user_id = data.get('userId')
        consultation_id = data.get('consultationId')
        current_question = data.get('currentQuestion')
        answer = data.get('answer')
        system_prompt = data.get('systemPrompt')

        if not all([user_id, consultation_id, current_question, answer, system_prompt]):
            return jsonify({
                'error': 'Missing required fields'
            }), 400

        # Prepare the conversation context for AI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current Question: {current_question}\nUser's Answer: {answer}\n\nBased on this answer, please:\n1. Extract relevant events with dates\n2. Note any mentioned files or documents\n3. Identify the client's goals\n4. Suggest immediate tasks\n5. Analyze possible outcomes\n6. Provide the next relevant question to ask\n\nFormat your response as JSON with the following structure:\n{{\n  \"events\": [\"event1\", \"event2\"],\n  \"files\": [\"file1\", \"file2\"],\n  \"goals\": [\"goal1\", \"goal2\"],\n  \"tasks\": [\"task1\", \"task2\"],\n  \"possibleOutcome\": [\"outcome1\", \"outcome2\"],\n  \"nextQuestion\": \"your next question\",\n  \"interviewQnA\": {{\n    \"question\": \"{current_question}\",\n    \"answer\": \"{answer}\"\n  }}\n}}"}
        ]

        # Get AI response
        ai_response = get_response_from_ai_gpt_4_32k(messages)

        # Parse AI response
        try:
            response_data = json.loads(ai_response)
        except json.JSONDecodeError:
            return jsonify({
                'error': 'Invalid AI response format'
            }), 500

        # Ensure all required fields are present
        required_fields = ['events', 'files', 'goals', 'tasks', 'possibleOutcome', 'nextQuestion', 'interviewQnA']
        for field in required_fields:
            if field not in response_data:
                response_data[field] = []

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in process_consultation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/gpt/get_previous_question', methods=['POST'])
def get_previous_question():
    try:
        # Get request data
        data = request.json
        user_id = data.get('userId')
        consultation_id = data.get('consultationId')

        if not all([user_id, consultation_id]):
            return jsonify({
                'error': 'Missing required fields'
            }), 400

        # Prepare the conversation context for AI
        messages = [
            {"role": "system", "content": "You are an AI legal assistant helping with an initial consultation. Please provide the previous question based on the consultation context."},
            {"role": "user", "content": "Get the previous question for this consultation."}
        ]

        # Get AI response
        ai_response = get_response_from_ai_gpt_4_32k(messages)

        return jsonify({
            "question": "What legal matter brings you here today?",  # Default first question
            "previousAnswer": "",
            "notepadData": {
                "events": [],
                "files": [],
                "goals": [],
                "tasks": [],
                "possibleOutcome": []
            }
        })

    except Exception as e:
        logging.error(f"Error in get_previous_question: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/gpt/start_consultation', methods=['POST'])
def start_consultation():
    try:
        # Get request data
        data = request.json
        user_id = data.get('userId')
        consultation_id = data.get('consultationId')

        if not all([user_id, consultation_id]):
            return jsonify({
                'error': 'Missing required fields'
            }), 400

        # Return the first question to start the consultation
        return jsonify({
            "firstQuestion": "What legal matter brings you here today?"
        })

    except Exception as e:
        logging.error(f"Error in start_consultation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)