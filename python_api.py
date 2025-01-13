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
#CORS(app)  # Enable CORS for all routes

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
    
    # Get the incomplete word at cursor position
    text_before_cursor = current_text[:cursor_position]
    words = text_before_cursor.split()
    incomplete_word = words[-1] if words else ''
    
    if len(incomplete_word) < 2:  # Only suggest after 2 characters
        return jsonify({"suggestions": []})
        
    messages = [
        {"role": "system", "content": f"""You are a specialized legal assistant for a California civil law practice. Your role is to provide relevant word and phrase completions in the context of:

1. Family Law:
- Divorce proceedings, child custody, spousal support, visitation rights
- Domestic partnerships, prenuptial agreements
- Child support calculations and modifications

2. Housing Law:
- Landlord-tenant disputes
- Eviction proceedings
- Rental agreements and lease violations
- Property maintenance issues
- Fair housing violations

3. Contract Law:
- Contract drafting and review
- Breach of contract cases
- Service agreements
- Business contracts
- Settlement agreements

4. Employment Law:
- Workplace discrimination
- Wrongful termination
- Wage and hour disputes
- Employment contracts
- Workplace harassment
- Workers' compensation

5. Property Law:
- Real estate transactions
- Property boundary disputes
- Easement issues
- HOA disputes
- Property title issues

6. Civil Harassment:
- Restraining orders
- Civil harassment orders
- Workplace protection orders
- Elder abuse prevention

When suggesting completions, consider:
- Common legal terminology used in California courts
- Relevant California Civil Code sections
- Standard legal document phrases
- Common case citation formats
- Procedural terms used in civil litigation
- Names of relevant California court forms
- Standard legal pleading language

Provide concise, context-appropriate suggestions that would be useful in legal documentation, correspondence, and court filings.
Current context: {system_prompt}
Provide 3-5 brief, relevant suggestions to complete: "{incomplete_word}"
Format as a simple array of strings."""},
        {"role": "user", "content": f"""You are a specialized legal assistant for a California civil law practice. Your role is to provide relevant word and phrase completions in the context of:

1. Family Law:
- Divorce proceedings, child custody, spousal support, visitation rights
- Domestic partnerships, prenuptial agreements
- Child support calculations and modifications

2. Housing Law:
- Landlord-tenant disputes
- Eviction proceedings
- Rental agreements and lease violations
- Property maintenance issues
- Fair housing violations

3. Contract Law:
- Contract drafting and review
- Breach of contract cases
- Service agreements
- Business contracts
- Settlement agreements

4. Employment Law:
- Workplace discrimination
- Wrongful termination
- Wage and hour disputes
- Employment contracts
- Workplace harassment
- Workers' compensation

5. Property Law:
- Real estate transactions
- Property boundary disputes
- Easement issues
- HOA disputes
- Property title issues

6. Civil Harassment:
- Restraining orders
- Civil harassment orders
- Workplace protection orders
- Elder abuse prevention

When suggesting completions, consider:
- Common legal terminology used in California courts
- Relevant California Civil Code sections
- Standard legal document phrases
- Common case citation formats
- Procedural terms used in civil litigation
- Names of relevant California court forms
- Standard legal pleading language

Provide concise, context-appropriate suggestions that would be useful in legal documentation, correspondence, and court filings.
Current context: {system_prompt}
Provide 3-5 brief, relevant suggestions to complete: "{incomplete_word}"
Format as a simple array of strings."""}
    ]
    
    try:
        response = get_response_from_ai_gpt_4_32k(messages)
        # Clean up suggestions - remove quotes and extra spaces
        suggestions = [
            s.strip().strip('"').strip("'").strip()
            for s in response.replace('[', '').replace(']', '').split(',')
        ]
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
