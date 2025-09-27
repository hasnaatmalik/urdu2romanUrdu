#!/usr/bin/env python3
import subprocess
import sys
import os
import time

# ADD YOUR NGROK AUTH TOKEN HERE
NGROK_TOKEN = "33GtxhCCI9sLJh56dTQwZsi3yXJ_893y1o6cdotRegzh78Zcd"  # Replace with your actual token

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
    print("✅ Packages installed")

def launch_with_auth():
    try:
        # Set auth token
        if NGROK_TOKEN and NGROK_TOKEN != "your_token_here":
            os.environ["NGROK_AUTHTOKEN"] = NGROK_TOKEN
            print("✅ Ngrok auth token set")
        else:
            print("❌ Please add your ngrok auth token to the script")
            return None
            
        from pyngrok import ngrok
        public_url = ngrok.connect(8501)
        print(f"🌍 Public URL: {public_url}")
        
        subprocess.Popen([
            "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        
        return public_url
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    install_packages()
    url = launch_with_auth()
    if url:
        print(f"Access your app at: {url}")
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__ == "__main__":
    main()
