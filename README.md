# 🚗 MLOps Project: Used Car Price Predictor

This project implements an end-to-end MLOps pipeline to predict used car prices. It encompasses data preprocessing, model training, deployment using Flask, and orchestration with Apache Airflow.

---

## 📁 Project Structure
├── airflow/ # Airflow DAGs and configurations<br>
├── data/ # Store data that get from kaggle<br>
├── mlflow/ # tracking server artifacts and logs<br>
├── notebooks/ # Jupyter notebooks for EDA and model development<br>
├── src/ # Source code for python app<br>
├── requirements.txt # Python dependencies<br>
├── .gitignore # Git ignore file<br>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Galaxiiz7/MlopsProject_UsedCarPricePredictor.git
   cd MlopsProject_UsedCarPricePredictor
2. **Set up and start Airflow and MLflow:**
   ```bash
   cd .\airflow\
   docker-compose build
   docker-compose up
3. **Access Airflow UI:**
   Open your browser and go to: http://localhost:8080
4. **Access MLflow UI:**
   Open your browser and go to: http://localhost:5000
---
## 🧪 Usage

### 1. Data Exploration and Preprocessing

Use the Jupyter notebooks in the `notebooks/` directory for Exploratory Data Analysis (EDA) and data preprocessing.

### 2. Model Training

**Run the training scripts located in the `src/` directory. For example:**
```bash
   python src/train_model.py
```

### 3. Model Deployment with Fast API
To build and run the REST API using Docker:
```bash
   uvicorn src.app:app --host 0.0.0.0 --port 10000
```
Once the container is running, the API will be accessible at: http://localhost:5000
### 4. Orchestrate Pipeline with Airflow
  1. Place your DAG scripts in the airflow/dags/ directory.
  2. Start the Airflow webserver and scheduler:
  ```bash
   docker-compose build
   docker-compose up
  ```
  3. Open the Airflow web interface at: http://localhost:8080
     From there, you can trigger and monitor your ML workflow.
---

## 📦 API Endpoints

The Used Car Price Predictor exposes the following RESTful API endpoints using FastAPI.

---

### 🏠 `GET /`

**Description:**  
Root endpoint to verify the API is up and running.

**Response Example:**

```json
{
  "message": "Hello from MLOpsProject UsedCarPricePredictor"
}
```
### 🔍 GET /ping
Description:
Health check endpoint.

Response Example:
```json
{
  "status": "OK"
}
```
### 💰 POST /predict
Description:
Predict the price of a used car based on its features.

Request Body:
Send a JSON object with the following fields:
| Field         | Type   | Description                                     |
| ------------- | ------ | ----------------------------------------------- |
| manufacturer  | string | Car manufacturer name                           |
| condition     | string | Condition of the car (e.g., "good", "like new") |
| cylinders     | string | Number of cylinders (e.g., "4 cylinders")       |
| fuel          | string | Type of fuel (e.g., "gas", "diesel")            |
| odometer      | float  | Distance driven in miles                        |
| title\_status | string | Title status (e.g., "clean", "salvage")         |
| transmission  | string | Type of transmission (e.g., "automatic")        |
| drive         | string | Drive type (e.g., "fwd", "rwd")                 |
| paint\_color  | string | Color of the car                                |
| type          | string | Type of the car (e.g., "SUV")                   |
| car\_age      | int    | Age of the car in years                         |

Example Request:
```json
{
  "manufacturer": "toyota",
  "condition": "good",
  "cylinders": "4 cylinders",
  "fuel": "gas",
  "odometer": 85000.0,
  "title_status": "clean",
  "transmission": "automatic",
  "drive": "fwd",
  "paint_color": "white",
  "type": "sedan",
  "car_age": 5
}
```
Success Response:
```json
{
  "predicted_price": 10350.75
}
```
Error Responses:

- 500 Internal Server Error if the model is not loaded or prediction fails:
  ```json
  {
  "detail": "Model not loaded"
  }
  ```
  or
   ```json
   {
  "detail": "Prediction error: <error_message>"
   }
   ```





