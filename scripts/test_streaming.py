import sys
import requests
import json

def test_streaming(file_path, url="http://localhost:11433/v1/audio/transcriptions"):
    print(f"Testing streaming transcription for: {file_path}")
    
    files = {
        'file': (file_path, open(file_path, 'rb'), 'audio/wav'),
    }
    data = {
        'stream': 'true',
    }

    try:
        response = requests.post(url, files=files, data=data, stream=True)
        response.raise_for_status()
        
        print("Incremental output:")
        print("-" * 20)
        
        full_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    content = decoded_line[6:]
                    if content == '[DONE]':
                        print("\n" + "-" * 20)
                        print("Stream finished.")
                        break
                    
                    try:
                        chunk = json.loads(content)
                        if 'text' in chunk:
                            new_text = chunk['text']
                            # Print only the new part if it's incremental, 
                            # or just print the whole thing to show progress
                            print(f"\r{new_text}", end="", flush=True)
                            full_text = new_text
                    except json.JSONDecodeError:
                        pass
        
        print(f"\n\nFinal Transcription: {full_text}")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_streaming.py <path_to_wav_file>")
        sys.exit(1)
    
    test_streaming(sys.argv[1])
