#!/usr/bin/env python3
"""
Universal Q&A system - works with ANY processed document
Uses the chunks created by pipeline_runner_qa_working.py
"""

import boto3
import json
import requests

def get_latest_processed_chunks():
    """Get the most recently processed document chunks from S3"""
    s3 = boto3.client('s3', region_name='us-east-1')
    
    # List all processed documents
    response = s3.list_objects_v2(
        Bucket='blackletter-files-prod',
        Prefix='summaries/',
        MaxKeys=1000
    )
    
    if 'Contents' not in response:
        print("❌ No processed documents found")
        return None, None
    
    # Find the most recent one
    files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    
    # Look for chatbot_source.txt (contains full document)
    for file in files:
        if file['Key'].endswith('chatbot_source.txt'):
            print(f"📄 Found latest processed document: {file['Key']}")
            
            # Download the full text
            obj = s3.get_object(Bucket='blackletter-files-prod', Key=file['Key'])
            full_text = obj['Body'].read().decode('utf-8')
            
            # Extract document info from path
            path_parts = file['Key'].split('/')
            username = path_parts[1] if len(path_parts) > 1 else "unknown"
            
            return full_text, username
    
    print("❌ No chatbot_source.txt found")
    return None, None

def create_chunks(text, chunk_size=2000, overlap=200):
    """Create overlapping chunks for Q&A"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Add page estimation
        start_page = max(1, (i // 500) + 1)  # ~500 words per page
        end_page = max(1, ((i + len(chunk_words)) // 500) + 1)
        
        chunks.append({
            'text': chunk_text,
            'start_page': start_page,
            'end_page': end_page
        })
    
    return chunks

def find_relevant_chunks(question, chunks, top_k=5):
    """Find chunks most relevant to the question"""
    question_words = set(question.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk['text'].lower().split())
        score = len(question_words.intersection(chunk_words))
        
        # Boost for exact phrase matches
        if question.lower() in chunk['text'].lower():
            score += 10
        
        scored_chunks.append((score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    relevant = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
    
    return relevant

def ask_question(question, chunks):
    """Ask a question about the document"""
    print(f"\n❓ Question: {question}")
    
    # Find relevant chunks
    relevant_chunks = find_relevant_chunks(question, chunks)
    
    if not relevant_chunks:
        print("💬 Answer: No relevant information found in this document.")
        return
    
    # Combine context
    context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Build citations
    citations = []
    for chunk in relevant_chunks:
        if chunk['start_page'] == chunk['end_page']:
            citations.append(f"p.{chunk['start_page']}")
        else:
            citations.append(f"pp.{chunk['start_page']}-{chunk['end_page']}")
    
    # Create prompt
    prompt = f"""Based on this document:

{context}

Question: {question}

Answer specifically and clearly based on the document content:"""

    try:
        # Ask Ollama
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'llama3.1:8b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Factual answers
                    'num_predict': 1000
                }
            },
            timeout=30)
        
        if response.status_code == 200:
            answer = response.json()['response'].strip()
            print(f"💬 Answer: {answer}")
            print(f"📖 Sources: {', '.join(citations)}")
        else:
            print("❌ Error: AI service unavailable")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Universal document Q&A system"""
    print("🚀 Blackletter Universal Q&A System")
    print("=" * 50)
    
    # Get the latest processed document
    full_text, username = get_latest_processed_chunks()
    
    if not full_text:
        print("❌ No processed documents found. Run pipeline_runner_qa_working.py first!")
        return
    
    # Create chunks
    print(f"📦 Creating chunks...")
    chunks = create_chunks(full_text)
    print(f"✅ Ready with {len(chunks)} chunks from latest document")
    
    # Interactive Q&A
    print(f"\n💬 Ask questions about the document (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question.strip():
                continue
                
            ask_question(question, chunks)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
