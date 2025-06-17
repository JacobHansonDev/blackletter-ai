import requests
import json

def summarize_chunk(text, prompt="Summarize this legal document section clearly and professionally, focusing on key facts, dates, parties, and obligations:"):
    """
    Summarize text using local Ollama Llama model
    """
    try:
        # Prepare the full prompt
        full_prompt = f"{prompt}\n\n{text}"
        
        # Call local Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Error: No response from AI")
        else:
            return f"Error: API returned status {response.status_code}"
            
    except Exception as e:
        return f"Error calling AI: {str(e)}"

if __name__ == "__main__":
    # Test the summarizer
    test_text = "This is a contract between Company A and Company B for software development services lasting 6 months with payment of $10,000 per month."
    
    print("🧪 Testing summarizer...")
    summary = summarize_chunk(test_text)
    print(f"📄 Summary: {summary}")
