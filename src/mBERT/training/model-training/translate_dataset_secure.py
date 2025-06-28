"""
Secure and improved dataset translation script using OpenAI GPT models.

This script addresses the security and functionality issues in the original
translate_dataset.py while providing the same core functionality.

Key improvements:
- Secure API key handling via environment variables
- Configurable translation parameters
- Better error handling and retry logic
- Rate limiting to prevent API abuse
- Progress tracking and cost estimation
- Support for multiple target languages
"""

import os
import pandas as pd
import time
import logging
from openai import OpenAI
from tqdm import tqdm
from typing import Optional, List
from config import translation_config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SecureDatasetTranslator:
    """Secure dataset translator with proper error handling and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, target_language: str = "Spanish"):
        """
        Initialize the translator.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            target_language: Target language for translation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.target_language = target_language
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = translation_config.openai_model
        self.delay = translation_config.delay_between_requests
        
        # Translation statistics
        self.stats = {
            "total_requests": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "total_tokens_used": 0,
            "estimated_cost": 0.0
        }
    
    def translate_text(self, text: str, max_retries: int = 3) -> Optional[str]:
        """
        Translate text with retry logic and error handling.
        
        Args:
            text: Text to translate
            max_retries: Maximum number of retry attempts
            
        Returns:
            Translated text or None if translation failed
        """
        if not text or text.strip() == "":
            return text
        
        system_prompt = (
            f"You are a professional multilingual translator. "
            f"Translate the following text to {self.target_language}. "
            f"If the text doesn't have meaning in any language, leave it unchanged. "
            f"Preserve the original meaning and tone."
        )
        
        for attempt in range(max_retries):
            try:
                self.stats["total_requests"] += 1
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=len(text.split()) * 2 + 50,  # Estimate based on input length
                    temperature=0.3  # Lower temperature for more consistent translations
                )
                
                translated_text = response.choices[0].message.content
                
                # Update statistics
                self.stats["successful_translations"] += 1
                if hasattr(response, 'usage') and response.usage:
                    self.stats["total_tokens_used"] += response.usage.total_tokens
                    # Rough cost estimation (GPT-3.5-turbo pricing)
                    self.stats["estimated_cost"] += response.usage.total_tokens * 0.000002
                
                logger.debug(f"Successfully translated: {text[:50]}...")
                return translated_text
                
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                self.stats["failed_translations"] += 1
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.delay
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to translate after {max_retries} attempts: {text[:50]}...")
                    return None
        
        return None
    
    def translate_dataset(self, 
                         input_file: str, 
                         output_file: str, 
                         text_column: str = "text",
                         label_column: str = "label") -> bool:
        """
        Translate an entire dataset.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            text_column: Name of the text column to translate
            label_column: Name of the label column to preserve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load dataset
            logger.info(f"Loading dataset from {input_file}")
            df = pd.read_csv(input_file)
            
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")
            
            logger.info(f"Dataset loaded: {len(df)} rows")
            
            # Prepare translated data
            translated_rows = []
            start_time = time.time()
            
            # Progress bar
            for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Translating to {self.target_language}"):
                translated_text = self.translate_text(row[text_column])
                
                if translated_text is not None:
                    translated_rows.append({
                        text_column: translated_text,
                        label_column: row[label_column]
                    })
                else:
                    # Keep original text if translation failed
                    logger.warning(f"Keeping original text for row {index}")
                    translated_rows.append({
                        text_column: row[text_column],
                        label_column: row[label_column]
                    })
                
                # Rate limiting
                time.sleep(self.delay)
            
            # Save translated dataset
            translated_df = pd.DataFrame(translated_rows)
            translated_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Log statistics
            logger.info("=== Translation Complete ===")
            logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"Total requests: {self.stats['total_requests']}")
            logger.info(f"Successful translations: {self.stats['successful_translations']}")
            logger.info(f"Failed translations: {self.stats['failed_translations']}")
            logger.info(f"Success rate: {self.stats['successful_translations']/self.stats['total_requests']*100:.1f}%")
            logger.info(f"Total tokens used: {self.stats['total_tokens_used']}")
            logger.info(f"Estimated cost: ${self.stats['estimated_cost']:.4f}")
            logger.info(f"Output saved to: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset translation failed: {e}")
            return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate SMS dataset using OpenAI GPT")
    parser.add_argument("--input", default=str(translation_config.input_dataset), 
                       help="Input CSV file path")
    parser.add_argument("--output", default=str(translation_config.output_dataset),
                       help="Output CSV file path")
    parser.add_argument("--language", default=translation_config.target_language,
                       help="Target language for translation")
    parser.add_argument("--text-column", default="text",
                       help="Name of the text column")
    parser.add_argument("--label-column", default="label", 
                       help="Name of the label column")
    
    args = parser.parse_args()
    
    # Validate configuration
    if not translation_config.validate():
        logger.error("Configuration validation failed")
        return False
    
    # Create translator
    translator = SecureDatasetTranslator(target_language=args.language)
    
    # Confirm before starting (can be expensive)
    print(f"About to translate dataset: {args.input}")
    print(f"Target language: {args.language}")
    print(f"Output file: {args.output}")
    print("WARNING: This operation may incur OpenAI API costs!")
    
    if input("Continue? (y/N): ").lower() != 'y':
        print("Translation cancelled.")
        return False
    
    # Perform translation
    success = translator.translate_dataset(
        input_file=args.input,
        output_file=args.output,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)