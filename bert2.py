import torch
from transformers import pipeline
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_qa_pipeline(timeout_minutes=5):
    """Initialize pipeline with timeout and fallback options"""
    start_time = datetime.now()
    timeout = timedelta(minutes=timeout_minutes)
    
    # Try different models in order of preference
    models_to_try = [
        "distilbert-base-cased-distilled-squad",  # Smaller, faster
        "bert-large-uncased-whole-word-masking-finetuned-squad"  # Original
    ]
    
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to load model: {model_name}")
            
            # Check timeout
            if datetime.now() - start_time > timeout:
                logger.warning("Model loading timed out")
                return None
                
            pipe = pipeline(
                task="question-answering",
                model=model_name,
                device=-1  # Force CPU for stability
            )
            logger.info(f"Successfully loaded {model_name}")
            return pipe
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            continue
    
    return None

def main():
    logger.info("Starting QA system...")
    
    # First verify basic PyTorch functionality
    try:
        test_tensor = torch.rand(2, 2)
        logger.info(f"PyTorch test successful (random tensor: {test_tensor})")
    except Exception as e:
        logger.error(f"PyTorch test failed: {str(e)}")
        return
    
    qa_pipeline = initialize_qa_pipeline()
    
    if not qa_pipeline:
        logger.error("All model loading attempts failed")
        return
    
    context = """
    Photosynthesis is the process used by plants to convert light energy into chemical energy.
    The basic chemical reaction is: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂.
    This means plants take in carbon dioxide and water, and produce glucose and oxygen.
    """
    
    questions = [
        "What do plants produce during photosynthesis?",
        "What are the outputs of photosynthesis?",
        "What gas is released by plants?"
    ]
    
    for question in questions:
        try:
            logger.info(f"\nProcessing: {question}")
            result = qa_pipeline(question=question, context=context)
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['score']:.1%}")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    logger.info("Script starting")
    main()
    logger.info("Script completed")
