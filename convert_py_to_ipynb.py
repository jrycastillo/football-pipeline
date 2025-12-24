
import nbformat as nbf
import os
import re

input_file = "/home/ubuntu/football/pipeline_consolidated.py"
output_file = "/home/ubuntu/football/pipeline_consolidated.ipynb"

def convert():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    with open(input_file, 'r') as f:
        text = f.readlines()

    nb = nbf.v4.new_notebook()
    cells = []
    
    current_chunk = []
    
    # Header regex: # --- X. TITLE ---
    header_pattern = re.compile(r"^#\s+---\s+\d+\.")

    for line in text:
        # Check if line creates a new section
        if header_pattern.match(line):
            # If we have accumulated lines, push them as a cell
            if current_chunk:
                cell_content = "".join(current_chunk).strip()
                if cell_content:
                    cells.append(nbf.v4.new_code_cell(cell_content))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)
            
    # Append last chunk
    if current_chunk:
        cell_content = "".join(current_chunk).strip()
        if cell_content:
            cells.append(nbf.v4.new_code_cell(cell_content))

    nb['cells'] = cells
    
    with open(output_file, 'w') as f:
        nbf.write(nb, f)
        
    print(f"Successfully converted {input_file} to {output_file}")
    print(f"Total Cells: {len(cells)}")

if __name__ == "__main__":
    convert()
