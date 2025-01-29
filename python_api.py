from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import openai
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
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
import pdfplumber
import importlib, json
from PIL import Image
import pytesseract
import io
import requests
import tempfile
import mimetypes

# Create Flask app instance
app = Flask(__name__)
CORS(app, resources={
    r"/gpt/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost",
            "https://app.associateattorney.ai",
            "https://associateattorney.ai",
            # Add any other production domains you need
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

load_dotenv()  # This loads the variables from .env

OPENAI_AZURE_API_VERSION = os.getenv('OPENAI_AZURE_API_VERSION')
OPENAI_AZURE_API_BASE_URL = os.getenv('OPENAI_AZURE_API_BASE_URL')
OPENAI_AZURE_API_KEY = os.getenv('OPENAI_AZURE_API_KEY')
OPENAI_AZURE_API_ENGINE = os.getenv('OPENAI_AZURE_API_ENGINE')

def encode_image_to_base64(image_url):
    try:
        # Get the image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch image: Status code {response.status_code}")

        # Determine image type from URL or content
        content_type = response.headers.get('content-type')
        if not content_type:
            # Try to guess from URL
            guessed_type = mimetypes.guess_type(image_url)[0]
            content_type = guessed_type if guessed_type else 'image/jpeg'

        # Open and convert image using PIL
        image = Image.open(io.BytesIO(response.content))

        # Convert animated GIFs to first frame
        if content_type == 'image/gif' and getattr(image, 'is_animated', False):
            image.seek(0)  # Get first frame

        # Convert to RGB if necessary (for PNG with transparency)
        if image.mode in ('RGBA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image)
            image = background

        # Save to bytes with appropriate format
        buffered = io.BytesIO()
        save_format = {
            'image/jpeg': 'JPEG',
            'image/png': 'PNG',
            'image/gif': 'PNG',  # Convert GIF to PNG
            'image/webp': 'WEBP'
        }.get(content_type, 'JPEG')

        image.save(buffered, format=save_format, quality=85)
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Return with correct mime type
        return f"data:{content_type};base64,{encoded_image}"
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def get_response_from_ai_gpt_4_32k(messages):
    try:
        llm = AzureChatOpenAI(
            openai_api_version=OPENAI_AZURE_API_VERSION,
            azure_endpoint=OPENAI_AZURE_API_BASE_URL,
            azure_deployment=OPENAI_AZURE_API_ENGINE,
            openai_api_key=OPENAI_AZURE_API_KEY,
            temperature=0.7,
        )

        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if isinstance(content, list):
                message_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            message_content.append(item["text"])
                        elif item.get("type") == "image_url":
                            image_url = item["image_url"]

                            if image_url.lower().endswith('.txt'):
                                # Handle TXT files
                                try:
                                    response = requests.get(image_url)
                                    if response.status_code == 200:
                                        # Get text content and decode
                                        txt_content = response.content.decode('utf-8', errors='ignore')
                                        message_content.append(f"Get summery of the following text:\n{txt_content}")
                                except Exception as e:
                                    print(f"Error processing TXT file: {str(e)}")
                                    message_content.append(f"Error reading TXT file: {str(e)}")

                            if image_url.lower().endswith('.pdf'):
                                # Handle PDF using pdfplumber
                                try:
                                    response = requests.get(image_url)
                                    if response.status_code == 200:
                                        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                                            temp_file.write(response.content)
                                            temp_path = temp_file.name
 
                                        try:
                                            pdf_text = ""
                                            with pdfplumber.open(temp_path) as pdf:
                                                for page in pdf.pages:
                                                    pdf_text += page.extract_text() or ""
                                                    pdf_text += "\n\n"  # Add spacing between pages
                                            message_content.append(f"PDF Content:\n{pdf_text}")
                                        finally:
                                            os.unlink(temp_path)  # Clean up temp file
                                except Exception as e:
                                    print(f"Error processing PDF: {str(e)}")
                                    message_content.append(f"Error reading PDF: {str(e)}")
                            else:
                                # Handle image URL (unchanged)
                                if image_url.startswith(('http://', 'https://', 'data:')):
                                    if image_url.startswith('data:'):
                                        # Already in base64 format
                                        base64_image = image_url
                                    else:
                                        # Convert URL to base64
                                        base64_image = encode_image_to_base64(image_url)

                                    if base64_image:
                                        message_content.append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": base64_image
                                            }
                                        })

                # Format final content (unchanged)
                final_content = []
                text_parts = []
                for item in message_content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    else:
                        if text_parts:
                            final_content.append({
                                "type": "text",
                                "text": "\n".join(text_parts)
                            })
                            text_parts = []
                        final_content.append(item)
                
                if text_parts:
                    final_content.append({
                        "type": "text",
                        "text": "\n".join(text_parts)
                    })

                if role == "user":
                    langchain_messages.append(HumanMessage(content=final_content))
                elif role == "system":
                    langchain_messages.append(SystemMessage(content=final_content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=final_content))
            else:
                # Handle regular text messages (unchanged)
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))

        response = llm(langchain_messages)
        return response.content

    except Exception as e:
        print(f"Azure OpenAI API Error: {str(e)}")
        return f"Error: {str(e)}"

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
    try:
        data = request.json
        prompt = data.get('prompt', {})
        system_prompt = data.get('systemPrompt')

        # Initialize messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Format user message based on prompt type
        if isinstance(prompt, dict) and 'image_url' in prompt:
            # If prompt has image_url, format content as array with text and image
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.get('text', '')},
                    {"type": "image_url", "image_url": prompt['image_url']}
                ]
            }
        else:
            # If no image_url, use prompt directly as content
            user_message = {
                "role": "user",
                "content": prompt
            }

        messages.append(user_message)
        
        response = get_response_from_ai_gpt_4_32k(messages)
        
        return jsonify({
            "response": response,
            "systemPrompt": system_prompt,
            "userPrompt": prompt
        })
        
    except Exception as e:
        print(f"Error in get_ai_response: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
            {"role": "user", "content": f"""Current Question: {current_question}
User's Answer: {answer}

Based on this answer, please:
1. Extract relevant events with dates
2. Note any mentioned files or documents
3. Identify the client's goals
4. Suggest immediate tasks
5. Analyze possible outcomes
6. Provide the next relevant question to ask

You must respond with valid JSON using this exact format:
{{
    "events": [],
    "files": [],
    "goals": [],
    "tasks": [],
    "possibleOutcome": [],
    "nextQuestion": "",
    "interviewQnA": {{
        "question": "{current_question}",
        "answer": "{answer}"
    }}
}}"""}
        ]

        # Get AI response
        ai_response = get_response_from_ai_gpt_4_32k(messages)

        # Try to parse AI response as JSON
        try:
            response_data = json.loads(ai_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to clean the response and parse again
            try:
                # Remove any markdown formatting or extra text
                cleaned_response = ai_response.strip()
                # Find the first { and last }
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    cleaned_json = cleaned_response[start_idx:end_idx]
                    response_data = json.loads(cleaned_json)
                else:
                    # If we can't find valid JSON, create a default response
                    response_data = {
                        "events": [],
                        "files": [],
                        "goals": [],
                        "tasks": [],
                        "possibleOutcome": [],
                        "nextQuestion": "Could you please clarify your previous response?",
                        "interviewQnA": {
                            "question": current_question,
                            "answer": answer
                        }
                    }
            except Exception:
                return jsonify({
                    'error': 'Invalid AI response format',
                    'raw_response': ai_response
                }), 500

        # Ensure all required fields are present
        required_fields = ['events', 'files', 'goals', 'tasks', 'possibleOutcome', 'nextQuestion', 'interviewQnA']
        for field in required_fields:
            if field not in response_data:
                response_data[field] = [] if field != 'nextQuestion' else ''
                if field == 'interviewQnA':
                    response_data[field] = {
                        "question": current_question,
                        "answer": answer
                    }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in process_consultation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/gpt/get_ai_response_v2', methods=['POST', 'OPTIONS'])
def get_ai_response_v2():
    try:
        if request.method == 'OPTIONS':
            response = jsonify({'message': 'OK'})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response

        data = request.json
        prompt = data.get('prompt', {})
        system_prompt = data.get('systemPrompt')
        
        # Template information text
        template_info = (
            "We have three types of templates: "
            "1. Letter template - For formal letter writing "
            "2. Legal template - For Legal writing "
            "3. General template - For general purpose writing "
            "Please provide appropriate template type and content based on the request. "
            "Response should be in JSON format with template_type "
            "template_content must be in letter format and not json array for letter template"
            "and not json object for general template."
            "template_content must be in legal format and not json array for legal template"
            "for attached files, template_type should be 'general template' and template_content should be the file meaning in text format."
        )

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Handle prompt based on type
        if isinstance(prompt, dict):
            if 'image_url' in prompt:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt.get('text', '')}\n\n{template_info}"},
                        {"type": "image_url", "image_url": prompt['image_url']}
                    ]
                })
            else:
                messages.append({
                    "role": "user", 
                    "content": f"{prompt.get('text', '')}\n\n{template_info}"
                })
        else:
            messages.append({
                "role": "user", 
                "content": f"{prompt}\n\n{template_info}"
            })

        # Get AI response
        ai_response = get_response_from_ai_gpt_4_32k(messages)

        try:
            # Initialize default response structure
            response_data = {
                'template_type': '',
                'template_content': '',
                'raw_response': ai_response,
                'systemPrompt': system_prompt,
                'userPrompt': prompt
            }

            # Clean and parse the raw response
            if isinstance(ai_response, str):
                cleaned_response = ai_response.strip()
                
                try:
                    parsed_response = json.loads(cleaned_response)
                    
                    if 'template_type' in parsed_response:
                        response_data['template_type'] = parsed_response['template_type']
                    
                    if 'template_content' in parsed_response:
                        content = parsed_response['template_content']
                        if isinstance(content, str):
                            response_data['template_content'] = content
                        elif isinstance(content, list):
                            response_data['template_content'] = content
                        else:
                            response_data['template_content'] = str(content)
                    
                    response_data['response'] = cleaned_response
                    
                except json.JSONDecodeError:
                    # Try to extract JSON from the response using regex
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
                    
                    if json_match:
                        try:
                            # Try parsing the extracted JSON
                            json_str = json_match.group(0)
                            parsed_response = json.loads(json_str)
                            
                            if 'template_type' in parsed_response:
                                response_data['template_type'] = parsed_response['template_type']
                            
                            if 'template_content' in parsed_response:
                                content = parsed_response['template_content']
                                if isinstance(content, str):
                                    response_data['template_content'] = content
                                elif isinstance(content, list):
                                    response_data['template_content'] = content
                                else:
                                    response_data['template_content'] = str(content)
                            
                            response_data['response'] = json_str
                            
                        except json.JSONDecodeError:
                            # If regex extraction fails, create a new JSON response
                            response_data['template_type'] = 'general template'
                            response_data['template_content'] = cleaned_response
                            response_data['response'] = cleaned_response
                    else:
                        # If no JSON-like structure found, treat as general template
                        response_data['template_type'] = 'general template'
                        response_data['template_content'] = cleaned_response
                        response_data['response'] = cleaned_response
            
            return jsonify(response_data)

        except Exception as parsing_error:
            print(f"Error parsing AI response: {str(parsing_error)}")
            # Create a fallback response
            return jsonify({
                'template_type': 'general template',
                'template_content': str(ai_response),
                'response': str(ai_response),
                'raw_response': str(ai_response),
                'systemPrompt': system_prompt,
                'userPrompt': prompt
            })

    except Exception as e:
        print(f"Error in get_ai_response_v2: {str(e)}")
        return jsonify({
            'error': str(e),
            'template_type': '',
            'template_content': '',
            'response': '',
            'raw_response': '',
            'systemPrompt': system_prompt,
            'userPrompt': prompt
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
        data = request.json
        user_id = data.get('userId')
        consultation_id = data.get('consultationId')
        user_data = data.get('userData', {})
        attorney_data = data.get('attorneyData', {})
        system_prompt = data.get('systemPrompt', '')

        if not all([user_id, consultation_id]):
            return jsonify({
                'error': 'Missing required fields'
            }), 400

        # Prepare the conversation context for AI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Generate a welcoming first message for the initial legal consultation.
Available information:
- Client Name: {user_data.get('name', '')}
- Client Email: {user_data.get('email', '')}
- Client Phone: {user_data.get('phone', '')}
- Preferred Language: {user_data.get('preferredLanguage', 'English')}
- Attorney Name: {attorney_data.get('name', 'Associate Attorney')}
- Attorney Specialization: {attorney_data.get('specialization', '')}
- Firm Name: {attorney_data.get('firmName', '')}

The message should:
1. Be warm and welcoming
2. Use the client's name if available
3. Briefly introduce the attorney/firm
4. Ask about their legal matter
5. Assure them of confidentiality

Return the message as a simple text string without any JSON formatting."""}
        ]

        # Get AI response
        welcome_message = get_response_from_ai_gpt_4_32k(messages)

        # Handle different response formats
        if isinstance(welcome_message, dict):
            # If it's a JSON response, extract just the message
            if 'message' in welcome_message:
                welcome_message = welcome_message['message']
            elif 'interviewQnA' in welcome_message:
                welcome_message = welcome_message['interviewQnA'].get('answer', '')
            elif 'firstQuestion' in welcome_message:
                welcome_message = welcome_message['firstQuestion']

        # Ensure we have a string
        welcome_message = str(welcome_message).strip()

        return jsonify({
            "firstQuestion": welcome_message
        })

    except Exception as e:
        logging.error(f"Error in start_consultation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/gpt/start_case_consultation', methods=['POST'])
def start_case_consultation():
    try:
        data = request.json
        prompt = data.get('prompt', {})
        system_prompt = data.get('systemPrompt')
        user_data = data.get('userData', {})

        # Prepare the conversation context for AI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Generate a chat response for the case detail of the case.

Available information:
- Client Name: {user_data.get('name', '')}
- Client Email: {user_data.get('email', '')}
- Client Phone: {user_data.get('phone', '')}  
The message should:
1. Be warm and welcoming
2. Use the client's name if available
2. Be in the language US accent.
5. Assure them of confidentiality


Return the message as a simple text string without any JSON formatting."""}
        ]

        # Get AI response
        chat_message = get_response_from_ai_gpt_4_32k(messages)
        # Ensure we have a string
        chat_message = str(chat_message).strip()

        return jsonify({"response": chat_message})

    except Exception as e:
        logging.error(f"Error in start_consultation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
