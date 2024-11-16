FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Install the required packages
COPY environment.yaml /app
RUN apt update && apt install -y ffmpeg
RUN conda env create -f /app/environment.yaml

# Trigger WhisperX to download necessary models, so we can embed them in the image
ARG hftoken
COPY audio_zh.wav audio_en.mp3 /app
RUN /bin/bash -c "source activate whisperx && cd /app; whisperx --hf_token $hftoken --model large-v3 --diarize --compute_type float32 --lang en ./audio_en.mp3"
RUN /bin/bash -c "source activate whisperx && cd /app; whisperx --hf_token $hftoken --model large-v3 --diarize --compute_type float32 --lang zh ./audio_zh.wav"

# Copy the app itself
RUN mkdir -p /app/uploads
COPY app.py config.py /app
#COPY whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask to run on 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0

# Run the command to start your app
CMD ["/opt/conda/envs/whisperx/bin/gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "--timeout", "900", "app:app"]
