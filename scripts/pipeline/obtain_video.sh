#!/bin/bash
CSV_FILE="searched_videos.csv"
OUTPUT_DIR="${OUTPUT_DIR:-data/videos}"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: $CSV_FILE not found in the current directory!"
    exit 1
fi

echo "Extracting URLs from $CSV_FILE and downloading videos to $OUTPUT_DIR..."

# Use python's built-in csv module to reliably parse out the URLs (handles commas in titles safely)
python -c "
import csv
with open('$CSV_FILE', mode='r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if 'url' in row and row['url'].strip():
            print(row['url'].strip())
" | while read -r url; do
    echo "================================================="
    echo "Starting download for: $url"
    
    # Run the extraction python script for each URL
    conda run -n mtb python src/download.py --url "$url" --output-dir "$OUTPUT_DIR"
done

echo "All download tasks finished!"