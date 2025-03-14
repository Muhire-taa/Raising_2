This project implements a machine learning system to predict households at risk of not achieving a $2/day income target. The solution includes a complete MLOps pipeline with automated data processing, model training, monitoring, and API deployment.
Project Structure

raising/
└── interview_data*.csv    # Uploaded data files
 ├── risk_model_*.joblib  # Trained model
 |___model_pipeline .py
 |__production.py
 |__raising.ipynb
 |__retrain.py
