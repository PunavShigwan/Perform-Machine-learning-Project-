"""
Test script for Pushup API
Run this after starting the FastAPI server
"""
import requests
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("=" * 50)
    print("Testing Health Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_pushup_health():
    """Test the pushup-specific health endpoint"""
    print("\n" + "=" * 50)
    print("Testing Pushup Health Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/pushup/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n" + "=" * 50)
    print("Testing Root Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_pushup_analyze(video_path: str):
    """Test the pushup analysis endpoint with a video file"""
    print("\n" + "=" * 50)
    print("Testing Pushup Analysis Endpoint")
    print("=" * 50)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    print(f"Uploading video: {video_path}")
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': (os.path.basename(video_path), video_file, 'video/mp4')}
            data = {
                'min_form_score': 75,
                'model_name': 'GradientBoosting'
            }
            
            print("Sending request...")
            response = requests.post(
                f"{BASE_URL}/pushup/analyze",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for video processing
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Analysis Successful!")
                print(f"Filename: {result.get('filename')}")
                print(f"Pushups Counted: {result.get('pushups')}")
                print(f"Estimated Target: {result.get('estimated_target')}")
                print(f"Average Form Score: {result.get('average_form_score', 0):.2f}%")
                print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
                
                rep_details = result.get('rep_details', [])
                if rep_details:
                    print(f"\nRep Details ({len(rep_details)} reps):")
                    for rep in rep_details[:5]:  # Show first 5 reps
                        print(f"  Rep {rep['rep_number']}: Form={rep['form_score']:.1f}%, "
                              f"Duration={rep.get('duration', 0):.2f}s")
                    if len(rep_details) > 5:
                        print(f"  ... and {len(rep_details) - 5} more reps")
                
                return True
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Video processing may take longer.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def find_test_video():
    """Try to find a test video in common locations"""
    possible_paths = [
        Path("ML_Model/pushup_model/testing_video"),
        Path("serverSide/ML_Model/pushup_model/testing_video"),
        Path("../ML_Model/pushup_model/testing_video"),
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            videos = list(base_path.glob("*.mp4"))
            if videos:
                return str(videos[0])
    
    return None

if __name__ == "__main__":
    print("🚀 Pushup API Test Script")
    print("=" * 50)
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Start it with: uvicorn app.main:app --reload")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_endpoint()
    pushup_health_ok = test_pushup_health()
    root_ok = test_root_endpoint()
    
    if not (health_ok and pushup_health_ok and root_ok):
        print("\n❌ Basic endpoints failed. Is the server running?")
        print("Start the server with: uvicorn app.main:app --reload")
        exit(1)
    
    # Test video analysis
    print("\n" + "=" * 50)
    print("Video Analysis Test")
    print("=" * 50)
    
    # Try to find a test video
    test_video = find_test_video()
    
    if test_video:
        print(f"Found test video: {test_video}")
        test_pushup_analyze(test_video)
    else:
        print("No test video found automatically.")
        print("Please provide a video path to test the analysis endpoint.")
        print("\nUsage:")
        print("  python test_pushup_api.py <video_path>")
        print("\nOr modify this script to include your video path.")
        
        # Check if video path provided as argument
        import sys
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            test_pushup_analyze(video_path)
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)
