"""
Test script to validate the improved training functionality without running full training.
"""

import sys
import logging
from train_ots_improved import ModelTrainer

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_components():
    """Test training components without full training."""
    try:
        logger.info("Testing ModelTrainer initialization...")
        trainer = ModelTrainer()
        
        logger.info("Testing dataset loading...")
        train_df, val_df, test_df = trainer.load_and_prepare_data()
        logger.info(f"Dataset loaded successfully: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        logger.info("Testing model initialization...")
        trainer.initialize_model()
        logger.info("Model initialized successfully")
        
        logger.info("Testing data loader creation...")
        train_loader = trainer.create_data_loader(train_df.head(10))  # Test with just 10 samples
        logger.info(f"Data loader created with {len(train_loader)} batches")
        
        logger.info("Testing single batch processing...")
        for batch in train_loader:
            logger.info(f"Batch keys: {batch.keys()}")
            logger.info(f"Input shape: {batch['input_ids'].shape}")
            logger.info(f"Labels shape: {batch['labels'].shape}")
            break  # Only test first batch
        
        logger.info("✅ All training components test successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_training_components()
    sys.exit(0 if success else 1)