import whisperx
import gc
import torch
import os
import json
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import config
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("faster_whisper")

audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
#default_compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
min_speakers=1
max_speakers=10
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {"mp3", "mp4", "wav", "awb", "aac", "ogg", "oga", "m4a", "wma", "amr"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # check if the post request has the file part
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    pretty_json = request.headers.get('X-Pretty-Json')
    compute_type = request.headers.get('X-Compute-Type')
    batch_size = request.headers.get('X-Batch-Size')
    language = request.headers.get('X-Lang')

    if not isinstance(batch_size, int) or batch_size > 256:
        batch_size = 16

    if compute_type.lower() not in ('float16', 'float32'):
        compute_type = 'int8'
        device = 'cpu'
        batch_size = 4
    else:
        device = 'cuda'

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = process(filepath, compute_type, device, batch_size, language)

        if pretty_json and pretty_json.lower() == 'true':
            # If the header is present and set to 'true', format JSON with 4 spaces indentation
            response = jsonify(result)
            response.data = json.dumps(response.json, indent=4)
            response.headers['Content-Type'] = 'application/json'
            return response
        else:
            # Default behavior if the header is not present or set to something other than 'true'
            return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def process(audio_file, compute_type, device, batch_size, language):
    logger.debug('Args passed to process transcription: compute_type: {0}, device: {1}, batch_size: {2}, language: {3}'.format(compute_type, device, batch_size, language))

    # Load models
    if language is None or language.lower() not in ('en', 'fr', 'de', 'es', 'it', 'ja', 'zh', 'nl', 'uk', 'pt'):
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    else:
        model = whisperx.load_model("large-v3", device, compute_type=compute_type, language=language)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=config.HF_TOKEN, device=device)

    # 1. Transcribe with original whisper (batched)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels
    diarize_segments = diarize_model(audio_file)

    # Return result
    result = whisperx.assign_word_speakers(diarize_segments, result)
    del model, diarize_model
    gc.collect
    torch.cuda.empty_cache()
    return result

if __name__ == '__main__':
    app.run(debug=True)
