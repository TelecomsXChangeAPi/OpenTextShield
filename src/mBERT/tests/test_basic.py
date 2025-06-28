#!/usr/bin/env python3
"""
Basic SMS classification test for OpenTextShield OTS-mBERT API.
"""

import time
from test_utils import TestRunner, OTSAPIClient


def main():
    """Test OpenTextShield OTS-mBERT API with basic SMS classification."""
    runner = TestRunner("Basic SMS Classification Test")
    
    if not runner.check_prerequisites():
        return False
    
    runner.start_test()
    
    # Sample text to classify
    sample_text = "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate). T&C's apply 08452810075over18's"
    
    print(f"Sample text: {sample_text[:100]}...")
    print()
    
    # Get prediction via API
    start_time = time.time()
    result = runner.client.predict(sample_text)
    request_time = time.time() - start_time
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        runner.logger.error(f"Test failed: {result['error']}")
        return False
    
    # Display results
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['probability']:.4f}")
    print(f"API Response Time: {result.get('processing_time', 'N/A')} seconds")
    print(f"Total Request Time: {request_time:.4f} seconds")
    
    # Performance feedback
    emoji = "😊" if request_time <= 1.0 else "😔"
    print(f"Performance: {emoji}")
    
    runner.process_prediction(result, sample_text)
    success = runner.finish_test(1)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)