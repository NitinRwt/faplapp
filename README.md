# FastAPI App

This is a FastAPI application that provides API services.

## 🚀 Features
- Fast and lightweight API
- Dockerized for easy deployment
- Uses Uvicorn as the ASGI server

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/NitinRwt/faplapp.git
cd faplapp
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the FastAPI application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 🐳 Running with Docker

### Pull the Docker image
```bash
docker pull sensei13k/faplapp
```
## OR

### Build the Docker image
```bash
docker build -t my-fastapi-app .
```

### Run the Docker container
```bash
docker run -p 8000:8000 my-fastapi-app
```

## 📌 API Endpoints
- **GET** `/` - Home route
- **POST** `/detect/` image process


Made with ❤️ by NitinRwt

