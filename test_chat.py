#!/usr/bin/env python3
import subprocess

# Load the document once
print("📄 Loading your document...")
with open('/home/ubuntu/extracted_text.txt', 'r') as f:
    document = f.read()

# Take first 3000 words so it fits in Ollama's context
words = document.split()
chunk = " ".join(words[:3000])

print(f"✅ Document loaded! ({len(chunk.split())} words)")
print()
print("🤖 You can now ask ANY question about this document!")
print("Type 'quit' to exit")
print("-" * 50)

while True:
    # Get question from user
    question = input("\n❓ Your question: ")
    
    if question.lower() == 'quit':
        print("👋 Goodbye!")
        break
    
    # Create prompt
    prompt = f"""Based on this document:

{chunk}

Question: {question}

Answer clearly and specifically based on what's in the document:"""
    
    # Send to Ollama
    print("\n🤖 Thinking...")
    try:
        result = subprocess.run(['ollama', 'run', 'llama3.1:8b'], 
                              input=prompt, 
                              text=True, 
                              capture_output=True)
        
        print(f"\n💬 Answer: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)
