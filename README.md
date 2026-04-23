-------------------------
Projekt structure:

<code>
homework_diabetes_risk_prediction/
│
├── app.py
├── db.py
├── config.py
├── ml_orchestrator.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── ml/
    ├── analysis.py
    ├── dataset.py
    ├── models.py
    ├── prediction.py
├── templates/
     ├── index.html
├── static/
     ├── style.css
├── data/  
     ├── diabetes.db
├── utilities
    ├── diabetes_dataset.py


Frontend (HTML/JS)
        ↓
Flask API (app.py)
        ↓
ML Orchestrator
        ↓
SQLite DB
        ↓
sklearn dataset

1 container = full stack
</code>

-------------------------
How to run:

Requirements:
    Docker installed
    Port 5000 free

Run from project root:
docker build -t diabetes-app .

The docker will:
    Build the image
    Install Python dependencies
    Copy the project repository

To run container:
docker run -p 5000:5000 diabetes-app

Run the application from browser:
    navigate to:
    http://localhost:5000

For handling persistent SQLite Data Table run container:
docker run -p 5000:5000 -v ${PWD}/data:/app/data diabetes-app

-------------------------
Features:

Frontend:
- Dataset summary (from SQLite DB)
- Visualizations (Chart.js)
- Diabetes risk prediction (ML model)
- Two threshold scenarios (150 / 250) ->
    Two threshold values are set hardcoded in config.py  
    Current threshold for App comes from config
- Flask REST API backend
- Machine Learning Orchestrator:
    One layer to handle ML scripts (analysis, dataset, models, prediction) and training for both scenarios (150, 250 thresholds)
- SQLite database:
    The application uses SQLite: data/diabetes.db
    It is automatically created and populated from: sklearn.datasets.load_diabetes()
