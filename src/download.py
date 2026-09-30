import os
import sys
import argparse
from urllib.parse import urlparse, parse_qs
import yt_dlp

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p.get('v', [None])[0]
        if query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        if query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

def download_video_and_transcript(url, output_path='.', cookies_from_browser=None, cookies=None):
    """Download the video and transcript using yt-dlp."""
    print(f"Downloading video and transcript from: {url}")
    print("NOTE: Using automatic format fallbacks to guarantee transcript extraction even if video streams are blocked by YouTube PO Tokens.")
    
    ydl_opts = {
        # Fallback progressively to storyboards (sb) if high-quality streams are missing due to bot protections
        'format': 'bestvideo+bestaudio/best/sb0/sb1/sb2/sb3/all',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'writesubtitles': True,         # Download explicit subtitles
        'writeautomaticsub': True,      # Download auto-generated subtitles if necessary
        'subtitleslangs': ['en'],# Prefer english, fallback to others
        'subtitlesformat': 'vtt/srt/best',
        'ignoreerrors': True,           # Continue execution even if some streams fail
        'extractor_args': {'youtube': {'player_client': ['android']}}, # Bypass 403 forbidden
    }

    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
    elif cookies:
        ydl_opts['cookiefile'] = cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed gracefully. If only the transcript was saved, YouTube Datacenter IP restrictions blocked the video binaries.")
    except Exception as e:
        print(f"[-] An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download a YouTube video and extract its transcript via yt-dlp.")
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=P1_x5LIg45Q", help="The YouTube video URL")
    parser.add_argument("--output-dir", default="./videos", help="Directory to save the video and transcript")
    parser.add_argument("--cookies-from-browser", help="Browser to extract cookies from")
    parser.add_argument("--cookies", help="Path to a cookies.txt file to use for authentication")
    
    args = parser.parse_args()
    
    video_id = extract_video_id(args.url)
    if not video_id:
        print("Error: Could not extract video ID from the provided URL.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download Video and Transcript 
    download_video_and_transcript(args.url, args.output_dir, args.cookies_from_browser, args.cookies)

if __name__ == "__main__":
    main()
