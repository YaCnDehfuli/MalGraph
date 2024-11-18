import sys
import io
import json
import pandas as pd
from contextlib import redirect_stdout

from volatility3.framework import contexts, interfaces
from volatility3.framework import automagic
from volatility3.cli.text_renderer import JsonRenderer
from volatility3.plugins import timeliner

def run_volatility_timeliner(memory_file, output_csv):
    # Initialize the context
    ctx = contexts.Context()

    # Register automagic modules
    automagics = automagic.available(ctx)

    # Set the single_location configuration to point to the memory dump file
    # Ensure that you use 'file:' as the prefix before the memory file path
    ctx.config['automagic:LayerStacker:single_location'] = f"file:{memory_file}"

    # Set the base configuration path for the Timeliner plugin
    base_config_path = interfaces.configuration.path_join("plugins", "timeliner")

    # Run Automagic to set up the necessary layers and configuration
    try:
        automagic.run(automagics, ctx, base_config_path)
    except Exception as e:
        print(f"Automagic failed: {e}")
        sys.exit(1)

    # Initialize and run the Timeliner plugin
    try:
        plugin = timeliner.Timeliner(ctx, base_config_path)
    except Exception as e:
        print(f"Failed to initialize the Timeliner plugin: {e}")
        sys.exit(1)

    # Run the plugin to get the output grid
    try:
        grid = plugin.run()
    except Exception as e:
        print(f"Failed to run the Timeliner plugin: {e}")
        sys.exit(1)

    # Initialize the JsonRenderer
    renderer = JsonRenderer()

    # Create an in-memory text stream to capture the JSON output
    output_buffer = io.StringIO()

    # Redirect stdout to the in-memory buffer to capture the JSON output
    with redirect_stdout(output_buffer):
        try:
            renderer.render(grid)
        except Exception as e:
            print(f"Failed to render JSON: {e}")
            sys.exit(1)

    # Retrieve the JSON string from the buffer
    json_output = output_buffer.getvalue()

    # Close the StringIO buffer
    output_buffer.close()

    # Parse the JSON data
    try:
        data = json.loads(json_output)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON output: {e}")
        sys.exit(1)

    # Validate the JSON structure
    if 'columns' not in data or 'rows' not in data:
        print("Invalid JSON structure: 'columns' or 'rows' key not found.")
        sys.exit(1)

    # Extract column names
    column_names = [column['name'] for column in data['columns']]

    # Extract row data
    rows = data['rows']

    # Convert the data to a pandas DataFrame
    try:
        df = pd.DataFrame(rows, columns=column_names)
    except Exception as e:
        print(f"Failed to create DataFrame: {e}")
        sys.exit(1)

    # Save the DataFrame to a CSV file
    try:
        df.to_csv(output_csv, index=False)
    except Exception as e:
        print(f"Failed to save DataFrame to CSV: {e}")
        sys.exit(1)

    print(f"Timeliner output successfully saved to '{output_csv}'.")


if __name__ == "__main__":
    # Specify the memory file and output CSV file paths
    memory_file_path = "/home/yacn/Research/dumps/wmplayer.raw"  # Replace with your memory dump file path
    output_csv_path = "/home/yacn/isos/timeliner_output.csv"     # Replace with your desired CSV output path

    # Run the Timeliner function
    run_volatility_timeliner(memory_file_path, output_csv_path)
