#!/usr/bin/env python3
"""
Script to help update import statements after project reorganization.
Run this script to get guidance on updating your import statements.
"""

import os
import sys

def main():
    print("🔄 Import Path Update Guide")
    print("=" * 50)
    print()
    
    print("After reorganization, update your import statements as follows:")
    print()
    
    print("📁 Data Processing Modules:")
    print("   Old: from preprocess import ...")
    print("   New: from src.data.preprocess import ...")
    print()
    print("   Old: from training_dataset import ...")
    print("   New: from src.data.training_dataset import ...")
    print()
    print("   Old: from combine_csv import ...")
    print("   New: from src.data.combine_csv import ...")
    print()
    
    print("🤖 Model Modules:")
    print("   Old: from rnn_models import ...")
    print("   New: from src.models.rnn_models import ...")
    print()
    
    print("📊 Data Paths:")
    print("   Raw data: data/raw/")
    print("   Processed data: data/processed/")
    print("   Models: models/")
    print()
    
    print("🚀 To run scripts:")
    print("   From project root: python scripts/main.py")
    print("   From project root: python scripts/training.py")
    print("   From project root: python scripts/backtest.py")
    print()
    
    print("💡 Make sure to run scripts from the project root directory")
    print("   so that Python can find the src/ package.")

if __name__ == "__main__":
    main()
