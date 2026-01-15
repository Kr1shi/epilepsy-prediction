#!/usr/bin/env python3
"""Simple script to peek into generated JSON files"""

import json
from pathlib import Path
from collections import Counter

def peek_json(filename):
    """Peek into a JSON file and display statistics"""
    
    filepath = Path(filename)
    if not filepath.exists():
        print(f"❌ File not found: {filename}")
        return
    
    print(f"\n{'='*70}")
    print(f"📄 FILE: {filename}")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if it's a sequences file
        if 'sequences' in data:
            sequences = data['sequences']
            print(f"\n✓ Total sequences: {len(sequences)}")
            
            if sequences:
                # Sample the first sequence
                first_seq = sequences[0]
                print(f"\n📋 Sample sequence (first one):")
                for key, value in first_seq.items():
                    if key in ['segment_starts', 'frequencies']:
                        if isinstance(value, list):
                            print(f"   {key}: [{len(value)} items] {value[:3]}...")
                        else:
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
                
                # Count by patient
                patients = Counter(seq.get('patient_id') for seq in sequences)
                print(f"\n👥 Sequences by patient:")
                for patient_id in sorted(patients.keys()):
                    print(f"   {patient_id}: {patients[patient_id]}")
                
                # Count by type
                types = Counter(seq.get('type') for seq in sequences)
                print(f"\n🏷️  Sequences by type:")
                for seq_type in sorted(types.keys()):
                    print(f"   {seq_type}: {types[seq_type]}")
                
                # Check for splits
                if 'split' in first_seq:
                    splits = Counter(seq.get('split') for seq in sequences)
                    print(f"\n✂️  Sequences by split:")
                    for split_name in sorted(splits.keys()):
                        print(f"   {split_name}: {splits[split_name]}")
            
            # Validation info
            if 'validation_info' in data:
                val_info = data['validation_info']
                print(f"\n🔍 Validation info:")
                for key, value in val_info.items():
                    print(f"   {key}: {value}")
        
        else:
            # Generic JSON file
            print(f"\n📊 JSON structure:")
            print(f"   Top-level keys: {list(data.keys())}")
            for key in data.keys():
                value = data[key]
                if isinstance(value, dict):
                    print(f"   {key}: {type(value).__name__} with {len(value)} items")
                elif isinstance(value, list):
                    print(f"   {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"   {key}: {value}")
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Try to find and display all JSON files
        json_files = list(Path('.').glob('*_sequences_*.json'))
        
        if json_files:
            print(f"Found {len(json_files)} sequence JSON files:\n")
            for f in sorted(json_files):
                print(f"  • {f}")
            
            print(f"\nPeeking into all files...\n")
            for json_file in sorted(json_files):
                peek_json(str(json_file))
        else:
            print("Usage: python peek_json.py <filename>")
            print("\nNo *_sequences_*.json files found in current directory.")
            print("\nExamples:")
            print("  python peek_json.py lopo_test_chb06_sequences_prediction.json")
            print("  python peek_json.py chb06_sequences_prediction.json")
            print("\nRun data_segmentation.py first to generate JSON files.")
    else:
        # Peek into specified file
        peek_json(sys.argv[1])
    
    print(f"\n{'='*70}\n")
