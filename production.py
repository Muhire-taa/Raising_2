
"""
Risk Assessment Model Production API

This script implements a FastAPI application that serves the risk assessment model.
It provides endpoints for making predictions and monitoring model performance.
"""

import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import shutil
import json

# Import the pipeline and monitoring classes
from model_pipeline import RiskAssessmentPipeline, ModelMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('risk_assessment_api')

# Initialize FastAPI application
app = FastAPI(
    title="Risk Assessment Model API",
    description="API for household risk assessment model predictions and monitoring",
    version="1.0.0"
)

# Global variables
pipeline = None
monitor = None
current_version = None

class PredictionRequest(BaseModel):
    """Request model for batch predictions."""
    data: List[Dict[str, Any]]

class MonitoringRequest(BaseModel):
    """Request model for monitoring results."""
    data_path: str
    true_labels_path: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global pipeline, monitor, current_version
    try:
        # Find the latest model version
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_files = [f for f in os.listdir(model_dir) if f.startswith('risk_model_')]
        if not model_files:
            logger.warning("No existing model found. Please train a model first.")
            return
        
        versions = [f.replace('risk_model_', '').replace('.joblib', '') for f in model_files]
        current_version = max(versions)
        
        # Load the model
        pipeline = RiskAssessmentPipeline()
        loaded = pipeline.load_model(current_version)
        
        if loaded:
            logger.info(f"Model version {current_version} loaded successfully")
            monitor = ModelMonitor(pipeline)
            
            # Load reference data if available
            try:
                reference_data_path = f"data/reference_data_{current_version}.csv"
                reference_data = pd.read_csv(reference_data_path)
                monitor.set_reference_data(reference_data)
                logger.info("Reference data loaded for monitoring")
            except FileNotFoundError:
                logger.warning("No reference data found for monitoring")
        else:
            logger.error(f"Failed to load model version {current_version}")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Risk Assessment Model API",
        "model_version": current_version,
        "status": "ready" if pipeline is not None else "no model loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "service unavailable", "message": "No model loaded"}
        )
    return {"status": "healthy", "model_version": current_version}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions for a batch of households."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = pipeline.predict(data)
        
        # Combine original data with predictions
        result = []
        for i, row in data.iterrows():
            result.append({
                "household_data": row.to_dict(),
                "prediction": {
                    "at_risk": bool(predictions['at_risk_prediction'].iloc[i]),
                    "risk_probability": float(predictions['risk_probability'].iloc[i])
                }
            })
        
        return {"predictions": result, "model_version": current_version}
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload data file for batch processing or retraining."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/uploaded_{timestamp}_{file.filename}"
        
        # Save the file
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": filename,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(file_path: str):
    """Make predictions for households in a CSV file."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Make predictions
        predictions = pipeline.predict(data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/batch_predictions_{timestamp}.csv"
        os.makedirs("results", exist_ok=True)
        
        # Combine original data with predictions
        result_df = pd.concat([data, predictions], axis=1)
        result_df.to_csv(output_path, index=False)
        
        return {
            "message": "Batch prediction completed",
            "output_file": output_path,
            "records_processed": len(data),
            "households_at_risk": int(predictions['at_risk_prediction'].sum()),
            "model_version": current_version
        }
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/monitor")
async def run_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks):
    """Run monitoring checks on new data."""
    if pipeline is None or monitor is None:
        raise HTTPException(status_code=503, detail="Model or monitor not initialized")
    
    try:
        # Check if file exists
        if not os.path.exists(request.data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Load data
        data = pd.read_csv(request.data_path)
        
        # Load true labels if provided
        true_labels = None
        if request.true_labels_path:
            if not os.path.exists(request.true_labels_path):
                raise HTTPException(status_code=404, detail=f"Labels file not found: {request.true_labels_path}")
            labels_df = pd.read_csv(request.true_labels_path)
            true_labels = labels_df['at_risk']
        
        # Run monitoring in background
        def run_monitoring_task():
            results = monitor.full_monitoring_check(data, true_labels)
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"metrics/monitoring_results_{timestamp}.json"
            os.makedirs("metrics", exist_ok=True)
            
            with open(output_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = json.dumps(results, default=lambda x: float(x) if hasattr(x, 'dtype') else x)
                f.write(json_results)
            
            logger.info(f"Monitoring results saved to {output_path}")
        
        background_tasks.add_task(run_monitoring_task)
        
        return {
            "message": "Monitoring check started in background",
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Monitoring error: {str(e)}")

@app.post("/retrain")
async def retrain_model(data_path: str, force: bool = False, background_tasks: BackgroundTasks):
    """Retrain the model with new data."""
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")
        
        # Run retraining in background
        def run_retraining_task():
            logger.info(f"Starting model retraining with data from {data_path}")
            
            # Create new pipeline and train model
            new_pipeline = RiskAssessmentPipeline()
            results = new_pipeline.run_pipeline(data_path)
            
            # Save new reference data
            data = pd.read_csv(data_path)
            data.to_csv(f"data/reference_data_{results['version']}.csv", index=False)
            
            logger.info(f"Model retrained successfully. New version: {results['version']}")
            
            # Update global variables
            global pipeline, monitor, current_version
            pipeline = new_pipeline
            monitor = ModelMonitor(pipeline)
            monitor.set_reference_data(data)
            current_version = results['version']
        
        background_tasks.add_task(run_retraining_task)
        
        return {
            "message": "Model retraining started in background",
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error in retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/versions")
async def list_versions():
    """List all available model versions."""
    try:
        model_dir = 'models'
        model_files = [f for f in os.listdir(model_dir) if f.startswith('risk_model_')]
        versions = [f.replace('risk_model_', '').replace('.joblib', '') for f in model_files]
        
        return {
            "versions": versions,
            "current_version": current_version
        }
    except Exception as e:
        logger.error(f"Error listing versions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing versions: {str(e)}")

@app.post("/switch-version/{version}")
async def switch_version(version: str):
    """Switch to a different model version."""
    global pipeline, monitor, current_version
    
    try:
        # Check if version exists
        model_path = f"models/risk_model_{version}.joblib"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model version not found: {version}")
        
        # Load the model
        new_pipeline = RiskAssessmentPipeline()
        loaded = new_pipeline.load_model(version)
        
        if loaded:
            pipeline = new_pipeline
            monitor = ModelMonitor(pipeline)
            current_version = version
            
            # Load reference data if available
            try:
                reference_data_path = f"data/reference_data_{version}.csv"
                reference_data = pd.read_csv(reference_data_path)
                monitor.set_reference_data(reference_data)
                logger.info("Reference data loaded for monitoring")
            except FileNotFoundError:
                logger.warning("No reference data found for monitoring")
            
            return {
                "message": f"Switched to model version {version}",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model version {version}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching versions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error switching versions: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)