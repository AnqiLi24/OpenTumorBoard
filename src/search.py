import yt_dlp
import csv

def get_youtube_ids(keyword="tumor board", min_duration_minutes=12, output_csv='searched_videos.csv'):
    """
    Collects YouTube video IDs that contain the specified keyword in the title
    and are longer than min_duration_minutes. Saves the results to a CSV file.
    """
    ydl_opts = {
        'extract_flat': True,
        'quiet': False, # Showing some progress is helpful for ytsearchall
        'ignoreerrors': True
    }
    
    videos = []
    min_duration_seconds = min_duration_minutes * 60
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Searching YouTube for '{keyword}' videos. This may take a moment...")
        # Search for all videos matching the keyword.
        # ytsearchall retrieves all available pages for the search query.
        info = ydl.extract_info(f'ytsearchall:"{keyword}"', download=False)
        
        if info and 'entries' in info:
            for entry in info['entries']:
                if not entry:
                    continue
                
                title = entry.get('title', '')
                duration = entry.get('duration')
                video_id = entry.get('id', '')
                
                if not video_id:
                    continue
                
                # Check if title contains the keyword (case-insensitive)
                # and duration is longer than the minimum
                if keyword.lower() in title.lower() and duration is not None and duration > min_duration_seconds:
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'duration': duration,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'title', 'duration', 'url'])
        writer.writeheader()
        writer.writerows(videos)
        
    print(f"Successfully filtered and saved {len(videos)} videos to {output_csv}")
    return output_csv

if __name__ == "__main__":
    get_youtube_ids(keyword="tumor board", min_duration_minutes=12)
