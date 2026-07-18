import os
from agent_processor import analyze_multimodal_inputs

def run_agent_loop():
    print("=" * 60)
    print("🤖 Multimodal AI Agent Active (Text, Speech, & Images)")
    print("=" * 60)
    print("Type your prompt. Optional: append --image <path> and/or --audio <path>")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down agent. Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Parsing primitive flags from standard terminal input
            parts = user_input.split(" ")
            text_prompt_list = []
            image_path = None
            audio_path = None
            
            i = 0
            while i < len(parts):
                if parts[i] == "--image" and i + 1 < len(parts):
                    image_path = parts[i+1]
                    i += 2
                elif parts[i] == "--audio" and i + 1 < len(parts):
                    audio_path = parts[i+1]
                    i += 2
                else:
                    text_prompt_list.append(parts[i])
                    i += 1
                    
            text_prompt = " ".join(text_prompt_list)
            
            # Fallback if user only provided files without instructions
            if not text_prompt:
                text_prompt = "Analyze the provided media files comprehensively."
                
            # Validate local file paths if provided before pinging the API
            if image_path and not os.path.exists(image_path):
                print(f"❌ Error: Image file not found at '{image_path}'")
                continue
            if audio_path and not os.path.exists(audio_path):
                print(f"❌ Error: Audio file not found at '{audio_path}'")
                continue

            # Run the unified pipeline
            result = analyze_multimodal_inputs(
                text_prompt=text_prompt,
                image_path=image_path,
                audio_path=audio_path
            )
            
            print("\n🤖 Agent Response:")
            print("-" * 40)
            print(result)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ An error occurred during execution: {e}")

if __name__ == "__main__":
    # Ensure your API Key is set in your environment variables
    #export api_key="AQ.Ab8RN6KYXCk_RuNkU-4C9PcGoAs5Z9tCmRk2103tBbaOfYHr5w"
    if "GEMINI_API_KEY" not in os.environ:
        print("⚠️ Warning: GEMINI_API_KEY environment variable not detected.")
    run_agent_loop()