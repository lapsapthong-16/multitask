import json

def count_labeling_values():
    """
    Read flat1.json and count the occurrences of 'Y' and 'N' 
    in the 'include_for_labeling' field
    """
    try:
        # Read the JSON file
        with open('flat4.json', 'r', encoding='utf-8') as file:
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

if __name__ == "__main__":
    count_labeling_values()
