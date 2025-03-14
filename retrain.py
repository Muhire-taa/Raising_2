import os
import pandas as pd
import logging
import schedule
import time
from datetime import datetime
import argparse
from model_pipeline import RiskAssessmentPipeline, ModelMonitor

# Setup logging
logging.basicConfig(
    filename=f"logs/retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_retraining')

def load_latest_model():
    """Load the latest model version."""
    logger.info("Loading latest model")
    model_dir = 'models'
    
    # Find the latest model version
    model_files = [f for f in os.listdir(model_dir) if f.startswith('risk_model_')]
    if not model_files:
        logger.error("No existing model found")
        return None
    
    versions = [f.replace('risk_model_', '').replace('.joblib', '') for f in model_files]
    latest_version = max(versions)
    
    # Create and load pipeline
    pipeline = RiskAssessmentPipeline()
    loaded = pipeline.load_model(latest_version)
    
    if loaded:
        logger.info(f"Latest model version {latest_version} loaded successfully")
        return pipeline, latest_version
    else:
        logger.error("Failed to load latest model")
        return None

def evaluate_and_retrain(data_path, force_retrain=False):
    """Evaluate current model and retrain if necessary."""
    logger.info(f"Starting evaluation and retraining process with data from {data_path}")
    
    # Load latest model
    result = load_latest_model()
    if result is None:
        logger.info("No existing model found. Training new model.")
        pipeline = RiskAssessmentPipeline()
        pipeline.run_pipeline(data_path)
        return
    
    pipeline, current_version = result
    
    # Load and split new data
    new_data = pd.read_csv(data_path)
    
    # Setup monitor
    monitor = ModelMonitor(pipeline)
    
    # Get reference data (could be stored separately in production)
    try:
        reference_data_path = f"data/reference_data_{current_version}.csv"
        reference_data = pd.read_csv(reference_data_path)
        monitor.set_reference_data(reference_data)
    except FileNotFoundError:
        logger.warning("No reference data found. Using new data as reference.")
        # Split new data into reference and evaluation sets
        train_data = new_data.sample(frac=0.5, random_state=42)
        eval_data = new_data.drop(train_data.index)
        monitor.set_reference_data(train_data)
        new_data = eval_data  # Use remaining data for evaluation
    
    # Run monitoring checks
    monitoring_results = monitor.full_monitoring_check(new_data)
    
    # Determine if retraining is needed
    retrain_needed = monitoring_results['retraining_recommended'] or force_retrain
    
    if retrain_needed:
        logger.info("Retraining recommended. Starting retraining process.")
        
        # Create new pipeline and train model
        new_pipeline = RiskAssessmentPipeline()
        results = new_pipeline.run_pipeline(data_path)
        
        # Save new reference data
        new_data.to_csv(f"data/reference_data_{results['version']}.csv", index=False)
        
        logger.info(f"Model retrained successfully. New version: {results['version']}")
        logger.info(f"F1 score of new model: {results['f1_score']:.4f}")
        
        # Compare with previous model
        if 'f1_score' in monitoring_results.get('performance', {}):
            prev_f1 = monitoring_results['performance']['f1_score']
            logger.info(f"Previous model F1 score: {prev_f1:.4f}")
            improvement = results['f1_score'] - prev_f1
            logger.info(f"Performance improvement: {improvement:.4f}")
    else:
        logger.info("Current model is performing well. No retraining needed.")

def scheduled_retraining(data_folder='data', schedule_type='weekly'):
    """Schedule regular retraining jobs."""
    logger.info(f"Setting up {schedule_type} retraining schedule")
    
    def job():
        # Get latest data file
        data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and not f.startswith('reference_data_')]
        if not data_files:
            logger.error("No data files found for retraining")
            return
        
        latest_data = max(data_files, key=lambda x: os.path.getmtime(os.path.join(data_folder, x)))
        data_path = os.path.join(data_folder, latest_data)
        
        # Run evaluation and retraining
        evaluate_and_retrain(data_path)
    
    # Set schedule based on type
    if schedule_type == 'daily':
        schedule.every().day.at("02:00").do(job)  # Run at 2 AM every day
    elif schedule_type == 'weekly':
        schedule.every().monday.at("02:00").do(job)  # Run at 2 AM every Monday
    elif schedule_type == 'monthly':
        schedule.every(30).days.at("02:00").do(job)  # Run approximately monthly
    else:
        logger.error(f"Unknown schedule type: {schedule_type}")
        return
    
    logger.info(f"{schedule_type.capitalize()} retraining scheduled")
    
    # Run the scheduling loop
    while True:                   
        schedule.run_pending()
        time.sleep(3600)  # Check every hour

def manual_retraining(data_path, force=False):
    """Manually trigger the retraining process."""
    logger.info(f"Manual retraining triggered for data: {data_path}")
    evaluate_and_retrain(data_path, force_retrain=force)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Risk Assessment Model Retraining')
    parser.add_argument('--mode', choices=['manual', 'scheduled'], default='manual',
                        help='Retraining mode: manual or scheduled')
    parser.add_argument('--data', type=str, help='Path to the data file for retraining')
    parser.add_argument('--schedule', choices=['daily', 'weekly', 'monthly'], default='weekly',
                        help='Schedule frequency for automatic retraining')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of evaluation results')
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        if not args.data:
            logger.error("Data path must be provided for manual retraining")
            parser.print_help()
            exit(1)
        manual_retraining(args.data, args.force)
    elif args.mode == 'scheduled':
        scheduled_retraining(schedule_type=args.schedule)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        parser.print_help()
        exit(1)