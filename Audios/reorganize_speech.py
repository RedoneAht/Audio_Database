import os
import shutil
from pathlib import Path

# Define the mapping for speakers to their IDs
SPEAKER_MAPPING = {
    'Female': {
        'Marion_Seclin_35': 1,
        'Charlie_Danger_29': 2,
        'audio_book': 3
    },
    'Male': {
        'Frederic_Saldmann_72': 1,
        'Emmanuel_Macron_47': 2,
        'Audio_books': 3
    }
}

def extract_duration_from_filename(filename):
    """Extract duration (5, 10, 15) from test filename"""
    # Expected format: test-5_1.wav or test-10_1.wav or test-15_1.wav
    parts = filename.replace('.wav', '').split('-')
    if len(parts) > 1:
        duration_part = parts[1].split('_')[0]
        return duration_part + 's'
    return None

def reorganize_dataset(source_root, dest_root):
    """
    Reorganize speech dataset from old structure to new structure
    
    Args:
        source_root: Path to 'Frensh' folder with old structure
        dest_root: Path where new 'Frensh' folder will be created
    """
    source_path = Path(source_root)
    dest_path = Path(dest_root)
    
    # Create destination root
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Process each gender
    for gender in ['Female', 'Male']:
        gender_folder = source_path / gender
        
        if not gender_folder.exists():
            print(f"Warning: {gender_folder} not found, skipping...")
            continue
        
        # Determine gender code (H for male, F for female)
        gender_code = 'H' if gender == 'Male' else 'F'
        
        # Create train and test folders in destination
        train_dest = dest_path / gender / 'train'
        test_dest = dest_path / gender / 'test'
        train_dest.mkdir(parents=True, exist_ok=True)
        test_dest.mkdir(parents=True, exist_ok=True)
        
        # Process each speaker
        for speaker_folder in gender_folder.iterdir():
            if not speaker_folder.is_dir():
                continue
            
            speaker_name = speaker_folder.name
            speaker_id = SPEAKER_MAPPING[gender].get(speaker_name)
            
            if speaker_id is None:
                print(f"Warning: Unknown speaker {speaker_name}, skipping...")
                continue
            
            print(f"\nProcessing {gender}/{speaker_name} (ID: {speaker_id})...")
            
            # Process train files
            train_folder = speaker_folder / 'train'
            if train_folder.exists():
                train_files = sorted([f for f in train_folder.iterdir() if f.suffix == '.wav'])
                for idx, train_file in enumerate(train_files, 1):
                    # New name: Fr_H_1_1.wav (for male) or Fr_F_1_1.wav (for female)
                    new_name = f"Fr_{gender_code}_{speaker_id}_{idx}.wav"
                    dest_file = train_dest / new_name
                    shutil.copy2(train_file, dest_file)
                    print(f"  Train: {train_file.name} -> {new_name}")
            
            # Process test files
            test_folder = speaker_folder / 'test'
            if test_folder.exists():
                test_files = sorted([f for f in test_folder.iterdir() if f.suffix == '.wav'])
                
                # Group files by duration
                files_by_duration = {}
                for test_file in test_files:
                    duration = extract_duration_from_filename(test_file.name)
                    if duration:
                        if duration not in files_by_duration:
                            files_by_duration[duration] = []
                        files_by_duration[duration].append(test_file)
                
                # Process each duration group
                for duration in sorted(files_by_duration.keys()):
                    files = files_by_duration[duration]
                    for idx, test_file in enumerate(files, 1):
                        # New name: Fr_h_1_10s_1.wav (for male) or Fr_f_1_10s_1.wav (for female)
                        new_name = f"Fr_{gender_code.lower()}_{speaker_id}_{duration}_{idx}.wav"
                        dest_file = test_dest / new_name
                        shutil.copy2(test_file, dest_file)
                        print(f"  Test: {test_file.name} -> {new_name}")
    
    print(f"\n✓ Reorganization complete! New structure created at: {dest_path}")
    print(f"\nSummary of new structure:")
    print(f"  {dest_path}/")
    print(f"    Male/train/ - Training files for male speakers")
    print(f"    Male/test/  - Test files for male speakers")
    print(f"    Female/train/ - Training files for female speakers")
    print(f"    Female/test/  - Test files for female speakers")

def preview_changes(source_root):
    """Preview what changes will be made without actually copying files"""
    print("PREVIEW MODE - No files will be moved\n")
    print("=" * 60)
    
    source_path = Path(source_root)
    
    for gender in ['Female', 'Male']:
        gender_folder = source_path / gender
        
        if not gender_folder.exists():
            continue
        
        gender_code = 'H' if gender == 'male' else 'F'
        
        for speaker_folder in gender_folder.iterdir():
            if not speaker_folder.is_dir():
                continue
            
            speaker_name = speaker_folder.name
            speaker_id = SPEAKER_MAPPING[gender].get(speaker_name)
            
            if speaker_id is None:
                continue
            
            print(f"\n{gender.upper()}/{speaker_name} (ID: {speaker_id})")
            print("-" * 60)
            
            # Preview train files
            train_folder = speaker_folder / 'train'
            if train_folder.exists():
                train_files = list(train_folder.glob('*.wav'))
                print(f"  Train files: {len(train_files)}")
                if train_files:
                    print(f"    Example: {train_files[0].name} -> Fr_{gender_code}_{speaker_id}_1.wav")
            
            # Preview test files
            test_folder = speaker_folder / 'test'
            if test_folder.exists():
                test_files = list(test_folder.glob('*.wav'))
                print(f"  Test files: {len(test_files)}")
                if test_files:
                    duration = extract_duration_from_filename(test_files[0].name)
                    if duration:
                        print(f"    Example: {test_files[0].name} -> Fr_{gender_code.lower()}_{speaker_id}_{duration}_1.wav")

# Main execution
if __name__ == "__main__":
    # CONFIGURATION - Update these paths
    SOURCE_ROOT = "./French"  # Path to your current French folder
    DEST_ROOT = "./French_new"  # Path where new structure will be created
    
    print("Speech Dataset Reorganizer")
    print("=" * 60)
    
    # First, preview the changes
    print("\n1. PREVIEWING CHANGES...")
    preview_changes(SOURCE_ROOT)
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input("\nDo you want to proceed with reorganization? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\n2. REORGANIZING FILES...")
        reorganize_dataset(SOURCE_ROOT, DEST_ROOT)
    else:
        print("\nOperation cancelled.")