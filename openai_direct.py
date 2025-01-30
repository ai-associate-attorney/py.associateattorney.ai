from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import time
import tempfile
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)

app = Flask(__name__)
CORS(app, resources={
    r"/associate-attorney/*": {
        "origins": ["http://localhost:3000", "http://localhost", "https://app.associateattorney.ai"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

load_dotenv()  # Load environment variables

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def fetch_with_timeout(messages, timeout=8000):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can change this to the model you want to use
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        return response
    except Exception as error:
        raise error

@app.route('/associate-attorney/chat', methods=['POST', 'OPTIONS'])
def handler():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200

    if not OPENAI_API_KEY:
        return jsonify({'error': 'OpenAI configuration is not complete'}), 500

    temp_path = None
    file_response = None
    assistant = None

    try:
        # Get the PDF file and prompt from the request
        data = request.get_json()
        if not data or 'pdf_base64' not in data:
            return jsonify({'error': 'PDF file is required'}), 400

        pdf_base64 = data['pdf_base64']
        messages = data['messages']

        # Validate messages format
        if not isinstance(messages, list):
            return jsonify({'error': 'Messages must be an array'}), 400

        # Ensure we have both system and user messages
        has_system = False
        has_user = False
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return jsonify({'error': 'Invalid message format'}), 400
            if msg['role'] == 'system':
                has_system = True
            elif msg['role'] == 'user':
                has_user = True

        if not (has_system and has_user):
            return jsonify({'error': 'Both system and user messages are required'}), 400

        # Decode base64 PDF
        pdf_data = base64.b64decode(pdf_base64.split(',')[1] if ',' in pdf_base64 else pdf_base64)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Upload file
        file_response = client.files.create(
            file=open(temp_path, "rb"),
            purpose="assistants"
        )

        # Create assistant
        assistant = client.beta.assistants.create(
            model="gpt-4o",
            description="Legal document analysis assistant",
            tools=[{"type": "file_search"}],
            name="PDF assistant"
        )

        # Create thread
        thread = client.beta.threads.create()

        # Add message to thread with the file
        system_message = next(msg['content'] for msg in messages if msg['role'] == 'system')
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')
        prompt = system_message + '\n\n' + user_message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[
                Attachment(
                    file_id=file_response.id,
                    tools=[AttachmentToolFileSearch(type="file_search")]
                )
            ],
            content=prompt
        )

        # Run thread
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            timeout=300
        )

        if run.status != "completed":
            raise Exception(f"Run failed with status: {run.status}")

        # Get messages
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        messages_list = [message for message in messages]
        response_text = messages_list[0].content[0].text.value

        # Cleanup resources
        if temp_path:
            os.unlink(temp_path)
        if file_response:
            client.files.delete(file_response.id)
        if assistant:
            client.beta.assistants.delete(assistant.id)

        return jsonify({
            'content': response_text
        }), 200

    except Exception as error:
        # Cleanup resources in case of error
        if temp_path:
            os.unlink(temp_path)
        if file_response:
            client.files.delete(file_response.id)
        if assistant:
            client.beta.assistants.delete(assistant.id)

        print('PDF processing error:', str(error))
        return jsonify({
            'error': 'Error processing PDF file',
            'details': str(error)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=True)
  