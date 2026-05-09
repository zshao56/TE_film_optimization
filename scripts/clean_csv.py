import os
import sys

# Get absolute path to the data directory based on the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
csv_path = os.path.join(project_root, 'data', 'simulations', 'metadata.csv')

def clean_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Scanning {file_path} for corrupted lines...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print("CSV is empty.")
        return

    # Extract header to know the expected number of commas
    header = lines[0]
    expected_fields = len(header.split(','))
    
    good_lines = [header]
    corrupted_count = 0

    for i, line in enumerate(lines[1:], start=2): # start=2 because index 0 is header (line 1), index 1 is line 2
        # Check if the line has the correct number of fields
        # Note: A valid line might have commas inside the JSON string.
        # We need a robust way to check. A simple and effective way for this specific crash 
        # is to check if the line starts with a valid 'sim_' prefix.
        
        if line.startswith('sim_'):
            # It looks like a valid line, let's keep it.
            # We can also add an extra check to ensure it doesn't end with a string of commas
            if not line.strip().endswith(',,,,,,,,,'):
                 good_lines.append(line)
            else:
                 print(f"Line {i} corrupted (trailing commas): {line[:50]}...")
                 corrupted_count += 1
        else:
            # If it doesn't start with 'sim_', it's definitely corrupted (like '9399765,,True...')
            print(f"Line {i} corrupted (bad prefix): {line[:50]}...")
            corrupted_count += 1

    # Overwrite the original file with only the good lines
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(good_lines)

    print("-" * 30)
    print(f"Scan complete. Removed {corrupted_count} corrupted lines.")
    print(f"The CSV now contains {len(good_lines) - 1} healthy records.")
    print("You can now safely resume your simulation.")

if __name__ == "__main__":
    clean_csv(csv_path)