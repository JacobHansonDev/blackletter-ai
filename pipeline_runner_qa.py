import boto3
import json
import traceback
from datetime import datetime
import uuid
import os
import sys
import requests
import PyPDF2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration - UPDATED FOR Q&A
MAX_WORKERS = min(20, os.cpu_count() * 2)
CHUNK_TIMEOUT = 60  # Reduced - no heavy processing per chunk
CHUNK_SIZE = 2000  # Smaller chunks for better Q&A (vs 50000 for summary)
CHUNK_OVERLAP = 200  # Word overlap between chunks
QA_TIMEOUT = 30  # Q&A response timeout

def get_instance_id():
    """Get EC2 instance ID with fallback"""
    try:
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=5)
        return response.text
    except:
        return "i-016606398a9831c5a"

def get_latest_upload():
    """Find the most recently uploaded file from S3"""
    s3_client = boto3.client('s3', region_name='us-east-1')
    response = s3_client.list_objects_v2(
        Bucket='blackletter-files-prod',
        Prefix='uploads/',
        MaxKeys=1000
    )
    
    if 'Contents' not in response:
        raise Exception("No files found in uploads folder")
    
    files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    s3_key = files[0]['Key']
    print(f"📥 Found latest upload: {s3_key}")
    return s3_key

def download_file_from_s3(s3_key):
    """Download file from S3 with retry logic"""
    s3_client = boto3.client('s3', region_name='us-east-1')
    filename = os.path.basename(s3_key)
    local_path = f"/home/ubuntu/{filename}"
    
    for attempt in range(3):
        try:
            s3_client.download_file('blackletter-files-prod', s3_key, local_path)
            print(f"📥 Downloaded: {local_path}")
            return local_path
        except Exception as e:
            if attempt < 2:
                print(f"⚠️ Download attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
            else:
                raise Exception(f"Failed to download file: {str(e)}")

def extract_username_from_path(s3_key):
    """Extract username from S3 path like uploads/username/file.pdf"""
    try:
        parts = s3_key.split('/')
        if len(parts) >= 2:
            return parts[1]  # Second part is username
        return "unknown"
    except:
        return "unknown"

def extract_pdf_text(pdf_path):
    """Extract text from PDF with improved error handling"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            page_count = len(pdf_reader.pages)
            print(f"📄 PDF has {page_count:,} pages")
            
            text = ""
            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"⚠️ Error extracting page {page_num + 1}: {str(e)}")
                    continue
            
            if not text.strip():
                raise Exception("No text could be extracted from PDF")
            
            print(f"✅ Extracted {len(text):,} characters from {page_count:,} pages")
            return text, page_count
            
    except Exception as e:
        raise Exception(f"Failed to extract PDF text: {str(e)}")

def chunk_document_for_qa(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Enhanced chunking for Q&A with overlap and metadata
    Returns chunks with page numbers and positional info
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Estimate page number (rough calculation)
        words_per_page = 500  # Average words per page
        start_page = max(1, i // words_per_page + 1)
        end_page = max(1, (i + len(chunk_words)) // words_per_page + 1)
        
        chunk_info = {
            'id': len(chunks),
            'text': chunk_text,
            'word_start': i,
            'word_end': i + len(chunk_words),
            'start_page': start_page,
            'end_page': end_page,
            'length': len(chunk_text)
        }
        
        chunks.append(chunk_info)
    
    print(f"📦 Created {len(chunks)} chunks with {overlap} word overlap")
    return chunks

def find_relevant_chunks(question, chunks, top_k=5):
    """
    Simple keyword-based relevance scoring
    TODO: Replace with vector search in Phase 2
    """
    question_words = set(question.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk['text'].lower().split())
        score = len(question_words.intersection(chunk_words))
        
        # Boost score for exact phrase matches
        if question.lower() in chunk['text'].lower():
            score += 10
        
        scored_chunks.append((score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    relevant_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
    
    print(f"🔍 Found {len(relevant_chunks)} relevant chunks for question")
    return relevant_chunks

def answer_question(question, chunks):
    """
    Answer a question using relevant document chunks
    """
    # Find relevant chunks
    relevant_chunks = find_relevant_chunks(question, chunks)
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the document to answer your question."
    
    # Combine relevant chunks
    context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Build sources citation
    page_ranges = []
    for chunk in relevant_chunks:
        if chunk['start_page'] == chunk['end_page']:
            page_ranges.append(f"p.{chunk['start_page']}")
        else:
            page_ranges.append(f"pp.{chunk['start_page']}-{chunk['end_page']}")
    
    sources = f"Sources: {', '.join(page_ranges)}"
    
    # Create prompt for Llama
    prompt = f"""Based on the following document sections, answer the question accurately and concisely.

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

Instructions:
- Answer based only on the information provided
- Be specific and cite details from the document
- If the information isn't in these sections, say so
- Keep your answer clear and professional

ANSWER:"""

    try:
        print(f"🤖 Processing question: {question}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for factual answers
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            },
            timeout=QA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "Error: No response from AI").strip()
            
            # Add sources to answer
            full_answer = f"{answer}\n\n{sources}"
            return full_answer
        else:
            return f"Error: AI service returned status {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def store_processed_document(job_id, chunks, full_text, document_info, username):
    """
    Store processed document chunks for Q&A
    For now, just save locally. Phase 2 will add vector database.
    """
    s3_client = boto3.client('s3', region_name='us-east-1')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = f"processed/{username}/{timestamp}_{job_id}"
    
    # Store chunks as JSON
    chunks_data = {
        'job_id': job_id,
        'document_info': document_info,
        'chunks': chunks,
        'processed_date': timestamp
    }
    
    chunks_key = f"{base_path}/chunks.json"
    s3_client.put_object(
        Bucket='blackletter-files-prod',
        Key=chunks_key,
        Body=json.dumps(chunks_data, indent=2),
        ContentType='application/json'
    )
    
    # Store full text for backup
    full_text_key = f"{base_path}/full_text.txt"
    s3_client.put_object(
        Bucket='blackletter-files-prod',
        Key=full_text_key,
        Body=full_text,
        ContentType='text/plain'
    )
    
    print(f"💾 Stored processed document: {base_path}")
    return base_path

def log_job_status(job_id, status, stage, username, error_message=None):
    """Log job status to DynamoDB"""
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('blackletter-jobs')
        
        item = {
            'job_id': job_id,
            'status': status,
            'stage': stage,
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'instance_id': get_instance_id()
        }
        
        if error_message:
            item['error_message'] = error_message
        
        table.put_item(Item=item)
        print(f"📊 Logged: {status} - {stage}")
        
    except Exception as e:
        print(f"⚠️ Failed to log status: {str(e)}")

def send_notification(job_id, status, message):
    """Send SNS notification"""
    try:
        sns = boto3.client('sns', region_name='us-east-1')
        
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:339712742264:blackletter-notifications',
            Message=message,
            Subject=f'Blackletter Q&A Processing {status.upper()}: {job_id}'
        )
        print(f"📧 Notification sent: {status}")
        
    except Exception as e:
        print(f"⚠️ Failed to send notification: {str(e)}")

def put_metric(metric_name, value, unit='Count'):
    """Put CloudWatch metric"""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
        
        cloudwatch.put_metric_data(
            Namespace='Blackletter/QA',
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Timestamp': datetime.now()
                }
            ]
        )
        
    except Exception as e:
        print(f"⚠️ Failed to put metric: {str(e)}")

def validate_environment():
    """Validate that required services are available"""
    print("🔍 Validating environment...")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama is running")
        else:
            raise Exception("Ollama not responding")
    except Exception as e:
        raise Exception(f"Ollama validation failed: {str(e)}")
    
    # Check AWS credentials
    try:
        boto3.client('s3').list_buckets()
        print("✅ AWS credentials valid")
    except Exception as e:
        raise Exception(f"AWS validation failed: {str(e)}")

def trigger_ec2_shutdown():
    """Schedule EC2 shutdown after processing"""
    try:
        instance_id = get_instance_id()
        
        # Create shutdown script
        shutdown_script = f"""#!/bin/bash
sleep 300  # Wait 5 minutes
aws ec2 stop-instances --instance-ids {instance_id} --region us-east-1
"""
        
        with open('/tmp/shutdown.sh', 'w') as f:
            f.write(shutdown_script)
        
        os.chmod('/tmp/shutdown.sh', 0o755)
        os.system('nohup /tmp/shutdown.sh > /tmp/shutdown.log 2>&1 &')
        
        print(f"⏰ Scheduled shutdown for instance {instance_id} in 5 minutes")
        
    except Exception as e:
        print(f"⚠️ Failed to schedule shutdown: {str(e)}")

def main():
    """Main Q&A processing pipeline"""
    job_id = str(uuid.uuid4())[:8]
    
    try:
        print(f"🚀 Starting Blackletter Q&A Processing - Job ID: {job_id}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Validate environment
        validate_environment()
        
        # Get document to process
        s3_key = get_latest_upload()
        username = extract_username_from_path(s3_key)
        log_job_status(job_id, 'started', 'download', username)
        
        # Download and extract
        local_file = download_file_from_s3(s3_key)
        filename = os.path.basename(local_file)
        
        print(f"\n{'='*50}")
        print("📄 DOCUMENT PROCESSING")
        print(f"{'='*50}")
        
        # Extract text
        log_job_status(job_id, 'processing', 'extraction', username)
        full_text, page_count = extract_pdf_text(local_file)
        
        document_info = {
            'filename': filename,
            'pages': page_count,
            'size': len(full_text),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Chunk for Q&A
        log_job_status(job_id, 'processing', 'chunking', username)
        chunks = chunk_document_for_qa(full_text)
        
        # Store processed document
        log_job_status(job_id, 'processing', 'storage', username)
        s3_path = store_processed_document(job_id, chunks, full_text, document_info, username)
        
        # Test Q&A functionality
        print(f"\n{'='*50}")
        print("🤖 TESTING Q&A FUNCTIONALITY")
        print(f"{'='*50}")
        
        test_question = "What is this document about?"
        test_answer = answer_question(test_question, chunks)
        print(f"❓ Test Question: {test_question}")
        print(f"🤖 Test Answer: {test_answer}")
        
        # Log completion
        log_job_status(job_id, 'completed', 'finished', username)
        
        # Metrics
        put_metric('DocumentsProcessed', 1)
        put_metric('DocumentPages', page_count)
        put_metric('ChunksCreated', len(chunks))
        
        # Success summary
        print(f"\n{'='*60}")
        print(f"🎉 Q&A PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📄 Document: {filename}")
        print(f"📊 Pages: {page_count:,}")
        print(f"📦 Chunks: {len(chunks):,}")
        print(f"💾 Stored: s3://blackletter-files-prod/{s3_path}")
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Send notification
        send_notification(job_id, 'completed',
            f'✅ Document ready for Q&A!\n\n'
            f'Document: {filename}\n'
            f'Pages: {page_count:,}\n'
            f'Chunks: {len(chunks):,}\n'
            f'Ready for questions!')
        
        # Schedule shutdown
        trigger_ec2_shutdown()
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ PROCESSING FAILED: {error_msg}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        
        log_job_status(job_id, 'failed', 'error', 'unknown', error_msg)
        send_notification(job_id, 'failed', f'❌ Processing failed: {error_msg}')
        
        # Still shutdown on failure
        trigger_ec2_shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()
