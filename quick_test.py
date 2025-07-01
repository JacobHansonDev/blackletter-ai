#!/usr/bin/env python3

# Super simple test: Can we answer questions about documents?

# Read the big document you already have
with open('/home/ubuntu/extracted_text.txt', 'r') as f:
    document = f.read()

# Take just the first 3000 words (manageable chunk)
words = document.split()
chunk = " ".join(words[:3000])

print("📄 Document chunk loaded!")
print(f"📊 Chunk size: {len(chunk.split())} words")
print()

# Test question
question = "What is this document about?"

print(f"❓ Question: {question}")
print()

# Create prompt for Ollama
prompt = f"""Based on this document:

{chunk}

Question: {question}

Answer briefly and clearly:"""

print("🤖 Asking Ollama...")
print()

# Save prompt to file so we can send it to Ollama
with open('test_prompt.txt', 'w') as f:
    f.write(prompt)

print("✅ Prompt saved to test_prompt.txt")
print()
print("Now run this command:")
print("ollama run llama3.1:8b < test_prompt.txt")
