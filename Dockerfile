FROM python:3.12
WORKDIR /docker 
COPY ["import.csv", "monitor.py", "requirements.txt", "train.py", "/docker/"]
RUN pip install -r /docker/requirements.txt && python train.py
CMD ["python", "./monitor.py", "ws://143.110.238.245:8000/stream", "meeting_request_classifier.pkl"]
