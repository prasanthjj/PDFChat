# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables to disable usage stats collection (to prevent write errors)
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_DISABLE_WATCHDOG_WARNINGS=true
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV HOME=/tmp

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
