FROM python:3.10-alpine

WORKDIR /app

# Install dependencies required for pip and other packages
RUN apk add --no-cache gcc musl-dev libffi-dev

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
