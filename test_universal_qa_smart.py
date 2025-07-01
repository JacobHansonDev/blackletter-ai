#!/usr/bin/env python3
"""
Universal Q&A system with SMART CHUNKING + SEMANTIC SEARCH
- Respects sentence boundaries
- Keeps related sections together
- Section-aware splitting for legal documents
- Optimized chunk overlap
"""

import boto3
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import re

# Initialize the embedding model
print("🤖 Loading semantic search model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
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

def smart_sentence_split(text):
    """
    Smart sentence splitting that handles legal document formatting
    Simplified version that avoids regex complexity
    """
    # Simple but effective sentence splitting
    # Split on periods, exclamation marks, question marks
    sentences = []
    
    # Basic split on sentence endings
    parts = re.split(r'[.!?]+\s+', text)
    
    for part in parts:
        part = part.strip()
        if len(part) > 20:  # Ignore very short fragments
            # Skip if it looks like a section number or abbreviation
            if not re.match(r'^\d+(\.\d+)*$', part):  # Not just numbers
                sentences.append(part)
    
    return sentences

def identify_section_boundaries(text):
    """
    Identify natural section boundaries in legal documents
    Returns positions where new sections likely start
    """
    boundaries = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Section headers patterns
        section_patterns = [
            r'^SEC\.\s*\d+',           # SEC. 123
            r'^SECTION\s*\d+',         # SECTION 123
            r'^\d+\.\s*[A-Z]',         # 1. TITLE
            r'^[A-Z]{2,}\s*[A-Z]',     # SUBTITLE A
            r'^\([a-z]\)\s*[A-Z]',     # (a) Content
            r'^\(\d+\)\s*[A-Z]',       # (1) Content
        ]
        
        if any(re.match(pattern, line) for pattern in section_patterns):
            # Calculate character position
            char_pos = sum(len(lines[j]) + 1 for j in range(i))  # +1 for newlines
            boundaries.append(char_pos)
    
    return boundaries

def create_smart_chunks(text, target_size=1800, max_size=2500, min_size=800):
    """
    Create smart chunks that respect document structure and sentence boundaries
    """
    print(f"📦 Creating smart chunks...")
    
    # First, identify section boundaries
    section_boundaries = identify_section_boundaries(text)
    print(f"🎯 Found {len(section_boundaries)} section boundaries")
    
    # Split into sentences
    sentences = smart_sentence_split(text)
    print(f"📝 Split into {len(sentences)} sentences")
    
    chunks = []
    current_chunk = ""
    current_start_pos = 0
    
    for sentence in sentences:
        # Calculate position in original text
        sentence_start = text.find(sentence, current_start_pos)
        
        # Check if adding this sentence would exceed target size
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Decision logic for chunk boundaries
        should_split = False
        
        # Split if we're over target size
        if len(potential_chunk) > target_size:
            should_split = True
        
        # Don't split if chunk would be too small
        if len(current_chunk) < min_size:
            should_split = False
        
        # Always split at section boundaries if chunk is reasonable size
        if (len(current_chunk) > min_size and 
            any(abs(sentence_start - boundary) < 100 for boundary in section_boundaries)):
            should_split = True
        
        # Never exceed max size
        if len(potential_chunk) > max_size:
            should_split = True
        
        if should_split and current_chunk:
            # Save current chunk
            chunk_start_page = max(1, current_start_pos // 2500 + 1)  # ~2500 chars per page
            chunk_end_page = max(1, (current_start_pos + len(current_chunk)) // 2500 + 1)
            
            chunks.append({
                'text': current_chunk.strip(),
                'start_page': chunk_start_page,
                'end_page': chunk_end_page,
                'char_start': current_start_pos,
                'char_end': current_start_pos + len(current_chunk),
                'sentence_count': len([s for s in current_chunk.split('.') if s.strip()])
            })
            
            # Start new chunk with overlap for context
            # Keep last 1-2 sentences for continuity
            overlap_sentences = current_chunk.split('.')[-2:]  # Last 2 sentences
            overlap_text = '. '.join([s.strip() for s in overlap_sentences if s.strip()])
            
            current_chunk = overlap_text + ". " + sentence if overlap_text else sentence
            current_start_pos = sentence_start - len(overlap_text) if overlap_text else sentence_start
        else:
            current_chunk = potential_chunk
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunk_start_page = max(1, current_start_pos // 2500 + 1)
        chunk_end_page = max(1, (current_start_pos + len(current_chunk)) // 2500 + 1)
        
        chunks.append({
            'text': current_chunk.strip(),
            'start_page': chunk_start_page,
            'end_page': chunk_end_page,
            'char_start': current_start_pos,
            'char_end': current_start_pos + len(current_chunk),
            'sentence_count': len([s for s in current_chunk.split('.') if s.strip()])
        })
    
    # Quality check
    avg_size = sum(len(chunk['text']) for chunk in chunks) / len(chunks)
    print(f"✅ Created {len(chunks)} smart chunks (avg: {avg_size:.0f} chars)")
    print(f"📊 Size range: {min(len(c['text']) for c in chunks)} - {max(len(c['text']) for c in chunks)} chars")
    
    return chunks

def create_chunks_with_smart_embeddings(text):
    """
    Create smart chunks and compute embeddings
    """
    # Create smart chunks
    chunks = create_smart_chunks(text)
    
    # Compute embeddings for all chunks at once
    print(f"🧠 Computing embeddings for {len(chunks)} smart chunks...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]
    
    print(f"✅ Smart chunks ready with semantic embeddings")
    return chunks

def find_relevant_chunks_smart(question, chunks, top_k=5):
    """
    Enhanced relevance finding with semantic similarity + structural hints
    """
    # Encode the question
    question_embedding = model.encode([question])
    
    # Calculate semantic similarity with structural bonuses
    similarities = []
    for chunk in chunks:
        # Base semantic similarity
        similarity = np.dot(question_embedding[0], chunk['embedding']) / (
            np.linalg.norm(question_embedding[0]) * np.linalg.norm(chunk['embedding'])
        )
        
        # Bonus for chunks with more sentences (likely more complete info)
        sentence_bonus = min(0.05, chunk['sentence_count'] * 0.01)
        
        # Bonus for chunks that contain section headers or numbered lists
        structure_bonus = 0.0
        text_lower = chunk['text'].lower()
        if any(pattern in text_lower for pattern in ['section', 'subsection', '(a)', '(b)', '(1)', '(2)']):
            structure_bonus = 0.02
        
        # Bonus for exact keyword matches (hybrid approach)
        question_words = set(question.lower().split())
        chunk_words = set(chunk['text'].lower().split())
        keyword_overlap = len(question_words.intersection(chunk_words)) / len(question_words)
        keyword_bonus = keyword_overlap * 0.03
        
        final_score = similarity + sentence_bonus + structure_bonus + keyword_bonus
        similarities.append((final_score, chunk, similarity))
    
    # Sort by final score
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Filter out very low similarity chunks
    relevant = [chunk for final_score, chunk, base_sim in similarities[:top_k] if base_sim > 0.1]
    
    # Debug: show similarity scores with bonuses
    print(f"🔍 Top smart matches:")
    for i, (final_score, chunk, base_sim) in enumerate(similarities[:3]):
        preview = chunk['text'][:100].replace('\n', ' ')
        bonus = final_score - base_sim
        print(f"  {i+1}. Score: {final_score:.3f} (base: {base_sim:.3f} +{bonus:.3f}) - {preview}...")
    
    return relevant

def ask_question_smart(question, chunks):
    """Ask a question using smart chunking + semantic search"""
    print(f"\n❓ Question: {question}")
    
    start_time = time.time()
    
    # Find relevant chunks using smart search
    relevant_chunks = find_relevant_chunks_smart(question, chunks)
    
    if not relevant_chunks:
        print("💬 Answer: No relevant information found in this document.")
        return
    
    # Combine context with better formatting
    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        context_parts.append(f"[Section {i+1}]\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Build enhanced citations
    citations = []
    for chunk in relevant_chunks:
        if chunk['start_page'] == chunk['end_page']:
            citations.append(f"p.{chunk['start_page']}")
        else:
            citations.append(f"pp.{chunk['start_page']}-{chunk['end_page']}")
    
    # Enhanced prompt for better answers
    prompt = f"""Based on these document sections, provide a comprehensive and accurate answer:

{context}

Question: {question}

Instructions:
- Use specific details and section references from the document
- If multiple sections provide related information, synthesize them
- Quote exact language when relevant
- Be comprehensive but concise

Answer:"""

    try:
        # Ask Ollama with optimized settings
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'llama3.1:8b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.05,  # Very factual
                    'top_p': 0.9,
                    'num_predict': 1200,  # More detailed answers
                    'repeat_penalty': 1.1
                }
            },
            timeout=45)  # Longer timeout for detailed answers
        
        search_time = time.time() - start_time
        
        if response.status_code == 200:
            answer = response.json()['response'].strip()
            print(f"💬 Answer ({search_time:.1f}s): {answer}")
            print(f"📖 Sources: {', '.join(citations)}")
        else:
            print(f"❌ Error: AI service returned {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Smart chunking + semantic Q&A system"""
    print("🚀 Blackletter SMART Q&A System")
    print("🧠 Smart Chunking + Semantic Search + Enhanced Retrieval")
    print("=" * 60)
    
    # Get the latest processed document
    full_text, username = get_latest_processed_chunks()
    
    if not full_text:
        print("❌ No processed documents found. Run pipeline_runner_qa_working.py first!")
        return
    
    # Create smart chunks with embeddings
    chunks = create_chunks_with_smart_embeddings(full_text)
    
    print(f"\n💬 Ask questions about the document (type 'quit' to exit):")
    print("🎯 Now using SMART CHUNKING for maximum accuracy!")
    print("-" * 60)
    
    # Test with challenging questions
    test_questions = [
        "What essential health benefits must all insurance plans cover under the ACA?",
        "How are subsidies for purchasing insurance calculated and distributed under the ACA?",
        "What penalties do employers face if they don't provide health insurance?",
        "How does the ACA protect people with pre-existing conditions?"
    ]
    
    print("\n🧪 TESTING WITH SMART CHUNKING:")
    print("=" * 60)
    
    for question in test_questions:
        ask_question_smart(question, chunks)
        print()
    
    # Interactive Q&A
    print("🤖 Interactive Smart Q&A:")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question.strip():
                continue
                
            ask_question_smart(question, chunks)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
