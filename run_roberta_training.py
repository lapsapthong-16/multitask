"""
Simple runner script for RoBERTa training pipeline
"""

import sys
import os
import argparse
from roberta_training import RobertaTrainer

def main():
    parser = argparse.ArgumentParser(description='Run RoBERTa training pipeline')
    parser.add_argument('--data', default='annotated_reddit_posts.csv', 
                       help='Path to Reddit dataset CSV file')
    parser.add_argument('--no-tuning', action='store_true', 
                       help='Skip hyperparameter tuning')
    parser.add_argument('--model', default='roberta-base', 
                       help='RoBERTa model to use (default: roberta-base)')
    parser.add_argument('--max-length', type=int, default=512, 
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found!")
        sys.exit(1)
    
    # Initialize trainer
    trainer = RobertaTrainer(model_name=args.model, max_length=args.max_length)
    
    # Run pipeline
    try:
        results = trainer.run_complete_pipeline(
            reddit_data_path=args.data,
            perform_hyperparameter_tuning=not args.no_tuning
        )
        
        print("\n✅ Training completed successfully!")
        print("Check the 'plots' and 'results' directories for outputs.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
