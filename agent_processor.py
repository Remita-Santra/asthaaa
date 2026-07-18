import time
from google import genai
from google.genai import types

# Initialize the modern unified client (automatically uses GEMINI_API_KEY from environment)
client = genai.Client()

def analyze_multimodal_inputs(
    text_prompt: str, 
    image_path: str = None, 
    audio_path: str = None, 
    model: str = "gemini-2.5-flash"
) -> str:
    """
    Processes complex text requests alongside optional images and audio 
    using the Gemini multimodal API.
    """
    # 1. Start with the core text query/instruction
    contents = [text_prompt]
    uploaded_files = []
    
    try:
        # 2. Handle Image Analysis if provided
        if image_path:
            print(f"[Agent] Uploading image: {image_path}...")
            img_file = client.files.upload(file=image_path)
            uploaded_files.append(img_file)
            contents.append(img_file)
            
        # 3. Handle Speech-to-Text & Audio Analysis if provided
        if audio_path:
            print(f"[Agent] Uploading audio: {audio_path}...")
            audio_file = client.files.upload(file=audio_path)
            uploaded_files.append(audio_file)
            
            # For longer audio, wait until the API finishes processing the file
            while audio_file.state.name == "PROCESSING":
                print("[Agent] Audio is still processing, waiting 2 seconds...")
                time.sleep(2)
                audio_file = client.files.get(name=audio_file.name)
                
            if audio_file.state.name == "FAILED":
                raise ValueError(f"Audio processing failed: {audio_file.error.message}")
                
            contents.append(audio_file)
            
        # 4. Execute the multimodal analysis
        print(f"[Agent] Routing inputs to {model}...")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            # Structured config to keep the agent tightly focused
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction="You are an elite analytical agent. Synthesize insights across text, spoken audio, and visual details perfectly."
            )
        )
        return response.text

    finally:
        # Clean up uploaded files from Gemini ephemeral storage (Good practice)
        for file in uploaded_files:
            try:
                client.files.delete(name=file.name)
                print(f"[Agent] Cleaned up file: {file.name}")
            except Exception as e:
                print(f"[Agent] Failed to delete file {file.name}: {e}")