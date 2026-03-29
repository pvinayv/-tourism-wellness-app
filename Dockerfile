FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860

# Force Streamlit to run on the port Hugging Face expects
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

LABEL build_date="2026-03-29 11:25:14"
# Forced update marker: 2026-03-29 11:25:14