"""
Dataset management utilities for OpenTextShield mBERT training.

Provides centralized dataset management without breaking existing functionality.
Organizes datasets by version and provides easy access to current and historical datasets.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages datasets for mBERT training with version control and organization."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.dataset_dir = self.base_dir / "dataset"
        
        # Ensure dataset directory exists
        self.dataset_dir.mkdir(exist_ok=True)
        
        # Current dataset (the one used by the API)
        self.current_dataset = "sms_spam_phishing_dataset_v2.1.csv"
        self.current_dataset_path = self.dataset_dir / self.current_dataset
    
    def list_datasets(self) -> List[Dict[str, str]]:
        """List all available datasets with metadata."""
        datasets = []
        
        for csv_file in sorted(self.dataset_dir.glob("*.csv")):
            try:
                # Get basic file info
                stat = csv_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(stat.st_mtime)
                
                # Try to get dataset info
                try:
                    df = pd.read_csv(csv_file)
                    row_count = len(df)
                    columns = list(df.columns)
                    
                    # Get label distribution if possible
                    label_dist = {}
                    if 'label' in df.columns:
                        label_dist = df['label'].value_counts().to_dict()
                
                except Exception as e:
                    logger.warning(f"Could not read dataset {csv_file.name}: {e}")
                    row_count = "Unknown"
                    columns = []
                    label_dist = {}
                
                datasets.append({
                    "filename": csv_file.name,
                    "path": str(csv_file),
                    "size_mb": round(size_mb, 2),
                    "modified": modified.strftime("%Y-%m-%d %H:%M:%S"),
                    "row_count": row_count,
                    "columns": columns,
                    "label_distribution": label_dist,
                    "is_current": csv_file.name == self.current_dataset
                })
                
            except Exception as e:
                logger.error(f"Error processing dataset {csv_file.name}: {e}")
                
        return datasets
    
    def get_current_dataset(self) -> Path:
        """Get path to the current dataset used by the API."""
        if not self.current_dataset_path.exists():
            # Try to find the latest dataset
            datasets = self.list_datasets()
            if datasets:
                latest = max(datasets, key=lambda x: x['modified'])
                self.current_dataset = latest['filename']
                self.current_dataset_path = Path(latest['path'])
                logger.info(f"Current dataset not found, using latest: {self.current_dataset}")
            else:
                raise FileNotFoundError("No datasets found in dataset directory")
        
        return self.current_dataset_path
    
    def get_dataset_info(self, filename: str) -> Optional[Dict[str, str]]:
        """Get detailed information about a specific dataset."""
        datasets = self.list_datasets()
        for dataset in datasets:
            if dataset['filename'] == filename:
                return dataset
        return None
    
    def validate_dataset(self, dataset_path: Path) -> Tuple[bool, List[str]]:
        """Validate dataset format and content."""
        issues = []
        
        try:
            if not dataset_path.exists():
                issues.append(f"Dataset file does not exist: {dataset_path}")
                return False, issues
            
            # Read dataset
            df = pd.read_csv(dataset_path)
            
            # Check required columns
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            # Check for empty dataset
            if len(df) == 0:
                issues.append("Dataset is empty")
            
            # Check for missing values
            if 'text' in df.columns:
                missing_text = df['text'].isna().sum()
                if missing_text > 0:
                    issues.append(f"Found {missing_text} missing text values")
            
            if 'label' in df.columns:
                missing_labels = df['label'].isna().sum()
                if missing_labels > 0:
                    issues.append(f"Found {missing_labels} missing label values")
                
                # Check valid labels
                valid_labels = {'ham', 'spam', 'phishing'}
                unique_labels = set(df['label'].dropna().unique())
                invalid_labels = unique_labels - valid_labels
                if invalid_labels:
                    issues.append(f"Found invalid labels: {invalid_labels}")
            
            # Check for duplicates
            if 'text' in df.columns:
                duplicates = df['text'].duplicated().sum()
                if duplicates > 0:
                    issues.append(f"Found {duplicates} duplicate text entries")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Error reading dataset: {e}")
            return False, issues
    
    def backup_dataset(self, dataset_name: str, backup_dir: Optional[Path] = None) -> Path:
        """Create a backup of a dataset."""
        if backup_dir is None:
            backup_dir = self.dataset_dir / "backups"
        
        backup_dir.mkdir(exist_ok=True)
        
        source_path = self.dataset_dir / dataset_name
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
        backup_path = backup_dir / backup_filename
        
        shutil.copy2(source_path, backup_path)
        logger.info(f"Dataset backed up to: {backup_path}")
        
        return backup_path
    
    def create_dataset_summary(self) -> str:
        """Create a summary report of all datasets."""
        datasets = self.list_datasets()
        
        if not datasets:
            return "No datasets found in dataset directory."
        
        summary = ["=== OpenTextShield Dataset Summary ===\n"]
        summary.append(f"Dataset directory: {self.dataset_dir}")
        summary.append(f"Total datasets: {len(datasets)}")
        summary.append(f"Current dataset: {self.current_dataset}\n")
        
        for dataset in datasets:
            summary.append(f"📄 {dataset['filename']}")
            summary.append(f"   Size: {dataset['size_mb']} MB")
            summary.append(f"   Rows: {dataset['row_count']}")
            summary.append(f"   Modified: {dataset['modified']}")
            
            if dataset['label_distribution']:
                summary.append(f"   Labels: {dataset['label_distribution']}")
            
            if dataset['is_current']:
                summary.append("   ⭐ CURRENT DATASET")
            
            summary.append("")
        
        return "\n".join(summary)
    
    def cleanup_old_datasets(self, keep_versions: int = 3) -> List[str]:
        """Clean up old dataset versions, keeping only the most recent ones."""
        datasets = self.list_datasets()
        
        # Filter only versioned datasets
        versioned_datasets = [
            d for d in datasets 
            if 'sms_spam_phishing_dataset_v' in d['filename']
        ]
        
        # Sort by version (assuming vX.Y format)
        versioned_datasets.sort(key=lambda x: x['filename'])
        
        # Keep only the latest versions
        if len(versioned_datasets) <= keep_versions:
            return []
        
        to_remove = versioned_datasets[:-keep_versions]
        removed_files = []
        
        for dataset in to_remove:
            # Don't remove the current dataset
            if dataset['is_current']:
                continue
            
            dataset_path = Path(dataset['path'])
            
            # Create backup before removing
            try:
                self.backup_dataset(dataset_path.name)
                dataset_path.unlink()
                removed_files.append(dataset['filename'])
                logger.info(f"Removed old dataset: {dataset['filename']}")
            except Exception as e:
                logger.error(f"Error removing dataset {dataset['filename']}: {e}")
        
        return removed_files


def main():
    """Command-line interface for dataset management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenTextShield Dataset Manager")
    parser.add_argument("action", choices=["list", "info", "validate", "summary", "cleanup"],
                       help="Action to perform")
    parser.add_argument("--dataset", help="Dataset filename for info/validate actions")
    parser.add_argument("--keep", type=int, default=3, help="Number of versions to keep during cleanup")
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.action == "list":
        datasets = manager.list_datasets()
        print(f"Found {len(datasets)} datasets:")
        for dataset in datasets:
            status = " (CURRENT)" if dataset['is_current'] else ""
            print(f"  {dataset['filename']}{status} - {dataset['size_mb']} MB - {dataset['row_count']} rows")
    
    elif args.action == "info":
        if not args.dataset:
            print("--dataset required for info action")
            return
        
        info = manager.get_dataset_info(args.dataset)
        if info:
            print(f"Dataset: {info['filename']}")
            print(f"Size: {info['size_mb']} MB")
            print(f"Rows: {info['row_count']}")
            print(f"Columns: {info['columns']}")
            print(f"Labels: {info['label_distribution']}")
            print(f"Modified: {info['modified']}")
        else:
            print(f"Dataset not found: {args.dataset}")
    
    elif args.action == "validate":
        if not args.dataset:
            dataset_path = manager.get_current_dataset()
        else:
            dataset_path = manager.dataset_dir / args.dataset
        
        is_valid, issues = manager.validate_dataset(dataset_path)
        print(f"Validating: {dataset_path.name}")
        
        if is_valid:
            print("✅ Dataset is valid")
        else:
            print("❌ Dataset has issues:")
            for issue in issues:
                print(f"  - {issue}")
    
    elif args.action == "summary":
        print(manager.create_dataset_summary())
    
    elif args.action == "cleanup":
        removed = manager.cleanup_old_datasets(keep_versions=args.keep)
        if removed:
            print(f"Removed {len(removed)} old datasets:")
            for filename in removed:
                print(f"  - {filename}")
        else:
            print("No old datasets to remove")


if __name__ == "__main__":
    main()