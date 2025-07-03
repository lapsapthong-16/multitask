import json
import os

def count_labeling_values():
    """
    Read flat1.json and count the occurrences of 'Y' and 'N' 
    in the 'include_for_labeling' field
    """
    try:
        # Read the JSON file
        with open('flat1.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Initialize counters
        count_y = 0
        count_n = 0
        count_empty = 0
        total_entries = len(data)
        
        # Count the values
        for entry in data:
            include_for_labeling = entry.get('include_for_labeling', '')
            
            if include_for_labeling == 'Y':
                count_y += 1
            elif include_for_labeling == 'N':
                count_n += 1
            else:
                count_empty += 1
        
        # Print results
        print("=== Include for Labeling Count ===")
        print(f"Total entries: {total_entries}")
        print(f"Include for labeling 'Y': {count_y}")
        print(f"Include for labeling 'N': {count_n}")
        print(f"Empty or other values: {count_empty}")
        print()
        print("=== Percentages ===")
        print(f"'Y' percentage: {(count_y/total_entries)*100:.2f}%")
        print(f"'N' percentage: {(count_n/total_entries)*100:.2f}%")
        print(f"Empty/other percentage: {(count_empty/total_entries)*100:.2f}%")
        
        return count_y, count_n, count_empty
        
    except FileNotFoundError:
        print("Error: flat1.json file not found!")
        return None, None, None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in flat1.json!")
        return None, None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None

def combine_y_labeled_data():
    """
    Read all flat*.json files and combine entries that have 'include_for_labeling': 'Y'
    """
    
    # List of files to process
    json_files = ['flat1.json', 'flat2.json', 'flat3.json', 'flat4.json', 'flat5.json']
    
    combined_data = []
    file_stats = {}
    
    for filename in json_files:
        try:
            if os.path.exists(filename):
                print(f"Processing {filename}...")
                
                with open(filename, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Filter for entries with 'include_for_labeling': 'Y'
                y_entries = [entry for entry in data if entry.get('include_for_labeling') == 'Y']
                
                # Add to combined data
                combined_data.extend(y_entries)
                
                # Track stats
                file_stats[filename] = {
                    'total_entries': len(data),
                    'y_entries': len(y_entries),
                    'percentage': (len(y_entries) / len(data)) * 100 if len(data) > 0 else 0
                }
                
                print(f"  - Total entries: {len(data)}")
                print(f"  - 'Y' labeled entries: {len(y_entries)}")
                print(f"  - Percentage: {file_stats[filename]['percentage']:.2f}%")
                
            else:
                print(f"Warning: {filename} not found, skipping...")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    total_entries = sum(stats['total_entries'] for stats in file_stats.values())
    total_y_entries = len(combined_data)
    
    print(f"Total entries across all files: {total_entries}")
    print(f"Total 'Y' labeled entries: {total_y_entries}")
    print(f"Overall percentage: {(total_y_entries/total_entries)*100:.2f}%" if total_entries > 0 else "No data processed")
    
    # Save combined data to a new file
    output_filename = 'combined_y_labeled_data.json'
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(combined_data, outfile, indent=2, ensure_ascii=False)
        print(f"\nCombined data saved to: {output_filename}")
        print(f"Combined file contains {len(combined_data)} entries")
        
    except Exception as e:
        print(f"Error saving combined data: {str(e)}")
    
    # Display breakdown by file
    print("\n" + "="*30)
    print("BREAKDOWN BY FILE")
    print("="*30)
    for filename, stats in file_stats.items():
        print(f"{filename}:")
        print(f"  Total: {stats['total_entries']}")
        print(f"  Y-labeled: {stats['y_entries']}")
        print(f"  Percentage: {stats['percentage']:.2f}%")
        print()
    
    return combined_data, file_stats

def analyze_combined_data(combined_data):
    """
    Analyze the combined Y-labeled data
    """
    if not combined_data:
        print("No data to analyze")
        return
    
    print("\n" + "="*30)
    print("DATA ANALYSIS")
    print("="*30)
    
    # Count by type
    type_counts = {}
    for entry in combined_data:
        entry_type = entry.get('type', 'unknown')
        type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
    
    print("Entries by type:")
    for entry_type, count in sorted(type_counts.items()):
        percentage = (count / len(combined_data)) * 100
        print(f"  {entry_type}: {count} ({percentage:.1f}%)")
    
    # Average scores
    scores = [entry.get('score', 0) for entry in combined_data if 'score' in entry]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\nAverage score: {avg_score:.2f}")
        print(f"Score range: {min(scores)} to {max(scores)}")
    
    # Average subjectivity
    subjectivities = [entry.get('subjectivity', 0) for entry in combined_data if 'subjectivity' in entry]
    if subjectivities:
        avg_subjectivity = sum(subjectivities) / len(subjectivities)
        print(f"Average subjectivity: {avg_subjectivity:.3f}")
        print(f"Subjectivity range: {min(subjectivities):.3f} to {max(subjectivities):.3f}")

if __name__ == "__main__":
    # count_labeling_values()

    print("Combining Y-labeled data from all flat*.json files...")
    print("="*60)
    
    combined_data, file_stats = combine_y_labeled_data()
    analyze_combined_data(combined_data)
    
    print("\n" + "="*60)
    print("Process completed!")
