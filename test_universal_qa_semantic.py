#!/usr/bin/env python3
"""
Universal Q&A system with SEMANTIC SEARCH using vector embeddings
Uses sentence-transformers for semantic similarity instead of keyword matching
"""

import boto3
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# Initialize the embedding model
print("🤖 Loading semantic search model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality model
print("✅ Semantic model loaded!")

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

def create_chunks_with_embeddings(text, chunk_size=1500, overlap=300):
    """
    Create overlapping chunks and compute embeddings for semantic search
    Smaller chunks = better precision for Q&A
    """
    words = text.split()
    chunks = []
    
    print(f"📦 Creating chunks with embeddings...")
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Add page estimation
        start_page = max(1, (i // 500) + 1)  # ~500 words per page
        end_page = max(1, ((i + len(chunk_words)) // 500) + 1)
        
        chunks.append({
            'text': chunk_text,
            'start_page': start_page,
            'end_page': end_page,
            'word_start': i,
            'word_end': i + len(chunk_words)
        })
    
    # Compute embeddings for all chunks at once (much faster)
    print(f"🧠 Computing embeddings for {len(chunks)} chunks...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]
    
    print(f"✅ Created {len(chunks)} chunks with semantic embeddings")
    return chunks

def find_relevant_chunks_semantic(question, chunks, top_k=5):
    """
    Find chunks using semantic similarity instead of keyword matching
    This is the key improvement!
    """
    # Encode the question
    question_embedding = model.encode([question])
    
    # Calculate semantic similarity with all chunks
    similarities = []
    for chunk in chunks:
        # Compute cosine similarity
        similarity = np.dot(question_embedding[0], chunk['embedding']) / (
            np.linalg.norm(question_embedding[0]) * np.linalg.norm(chunk['embedding'])
        )
        similarities.append((similarity, chunk))
    
    # Sort by similarity and return top chunks
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Filter out very low similarity chunks (< 0.1)
    relevant = [chunk for score, chunk in similarities[:top_k] if score > 0.1]
    
    # Debug: show similarity scores
    print(f"🔍 Top semantic matches:")
    for i, (score, chunk) in enumerate(similarities[:3]):
        preview = chunk['text'][:100].replace('\n', ' ')
        print(f"  {i+1}. Score: {score:.3f} - {preview}...")
    
    return relevant

def ask_question_semantic(question, chunks):
    """Ask a question using semantic search"""
    print(f"\n❓ Question: {question}")
    
    start_time = time.time()
    
    # Find relevant chunks using semantic similarity
    relevant_chunks = find_relevant_chunks_semantic(question, chunks)
    
    if not relevant_chunks:
        print("💬 Answer: No semantically relevant information found in this document.")
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
        
        search_time = time.time() - start_time
        
        if response.status_code == 200:
            answer = response.json()['response'].strip()
            print(f"💬 Answer ({search_time:.1f}s): {answer}")
            print(f"📖 Sources: {', '.join(citations)}")
        else:
            print("❌ Error: AI service unavailable")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Semantic Q&A system"""
    print("🚀 Blackletter Semantic Q&A System")
    print("=" * 50)
    
    # Get the latest processed document
    full_text, username = get_latest_processed_chunks()
    
    if not full_text:
        print("❌ No processed documents found. Run pipeline_runner_qa_working.py first!")
        return
    
    # Create chunks with embeddings
    chunks = create_chunks_with_embeddings(full_text)
    
    print(f"\n💬 Ask questions about the document (type 'quit' to exit):")
    print("🧠 Now using SEMANTIC SEARCH for better results!")
    print("-" * 50)
    
    # Test with the same questions that failed before
    test_questions = [
        "What essential health benefits must all insurance plans cover under the ACA?",
        "How are subsidies for purchasing insurance calculated and distributed under the ACA?"
    ]
    
    print("\n🧪 TESTING WITH PREVIOUSLY FAILED QUESTIONS:")
    print("=" * 50)
    
    for question in test_questions:
        ask_question_semantic(question, chunks)
        print()
    
    # Interactive Q&A
    print("🤖 Interactive Q&A (semantic search enabled):")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question.strip():
                continue
                
            ask_question_semantic(question, chunks)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
