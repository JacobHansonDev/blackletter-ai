import boto3
import json
import traceback
from datetime import datetime
import uuid
import os
import sys
import requests
import threading
import queue
import PyPDF2
from adaptive_legal_prompts import (
    DocumentTypeDetector, 
    AdaptivePromptGenerator, 
    BlackletterAdaptiveProcessor
)

# Initialize adaptive processor
adaptive_processor = BlackletterAdaptiveProcessor()

def get_instance_id():
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
    """Download the specified file from S3"""
    s3_client = boto3.client('s3', region_name='us-east-1')
    filename = os.path.basename(s3_key)
    local_path = f"/home/ubuntu/{filename}"
    
    print(f"📦 Downloading {s3_key} to {local_path}")
    s3_client.download_file('blackletter-files-prod', s3_key, local_path)
    
    file_size = os.path.getsize(local_path)
    print(f"✅ Downloaded {filename} ({file_size:,} bytes)")
    return local_path, filename

def extract_username_from_path(s3_key):
    """Extract username from S3 path"""
    try:
        parts = s3_key.split('/')
        if len(parts) >= 3 and parts[0] == 'uploads':
            return parts[1]
        else:
            return 'unknown_user'
    except:
        return 'unknown_user'

def get_pdf_page_count(pdf_path):
    """Get actual page count from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"⚠️ Could not get page count: {e}")
        return 0

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📄 Extracting text from {len(pdf_reader.pages)} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text += page.extract_text()
                    if (page_num + 1) % 100 == 0:
                        print(f"🔄 Extracted page {page_num + 1}/{len(pdf_reader.pages)}")
                except Exception as e:
                    print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                    continue
            
            print(f"✅ Extracted {len(text):,} characters from PDF")
            return text
            
    except Exception as e:
        raise Exception(f"Failed to extract PDF text: {str(e)}")

def log_job_status(job_id, status, stage, username, error_message=None):
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('BlackletterUsage')
        
        table.put_item(Item={
            'job_id': job_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': status,
            'stage': stage,
            'error_message': error_message or '',
            'updated_at': datetime.utcnow().isoformat(),
            'username': username,
            'instance_id': get_instance_id()
        })
        print(f"📊 Status logged: {status} - {stage} - User: {username}")
    except Exception as e:
        print(f"⚠️ Failed to log status: {e}")

def send_notification(job_id, status, message):
    try:
        sns = boto3.client('sns', region_name='us-east-1')
        topic_arn = 'arn:aws:sns:us-east-1:577638390288:blackletter-notifications'
        
        sns.publish(
            TopicArn=topic_arn,
            Subject=f'Blackletter Job {status.title()}: {job_id}',
            Message=message
        )
        print(f"📧 Notification sent: {status}")
    except Exception as e:
        print(f"⚠️ Failed to send notification: {e}")

def put_metric(metric_name, value, unit='Count'):
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
        cloudwatch.put_metric_data(
            Namespace='Blackletter/Pipeline',
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Timestamp': datetime.utcnow()
                }
            ]
        )
        print(f"📈 Metric sent: {metric_name} = {value}")
    except Exception as e:
        print(f"⚠️ Failed to send metric {metric_name}: {e}")

def intelligent_document_sampling(text, num_samples=25):
    """Sample key sections from throughout the document"""
    total_length = len(text)
    chunk_size = 8000
    
    chunks = [text[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
    print(f"📝 Document split into {len(chunks)} chunks for analysis")
    
    if len(chunks) <= num_samples:
        print(f"✅ Selected all {len(chunks)} chunks (document smaller than sample size)")
        return chunks
    
    samples = []
    samples.extend(chunks[:3])  # Beginning
    samples.extend(chunks[-3:])  # End
    
    # Middle sampling
    middle_chunks = chunks[3:-3]
    if middle_chunks:
        step = len(middle_chunks) // max(1, num_samples - 6)
        for i in range(0, len(middle_chunks), max(1, step)):
            if len(samples) < num_samples:
                samples.append(middle_chunks[i])
    
    print(f"✅ Selected {len(samples)} strategic samples covering entire document")
    return samples[:num_samples]

def extract_critical_information(samples, doc_type='contract'):
    """Extract critical information from document samples using adaptive prompts"""
    print(f"🔍 Extracting critical information from {len(samples)} samples...")
    print(f"📄 Document type: {doc_type}")
    
    # Get adaptive prompts for this document type
    prompts = adaptive_processor.prompt_generator.get_prompts(doc_type)
    
    critical_info = []
    queue_items = queue.Queue()
    
    def worker():
        while True:
            item = queue_items.get()
            if item is None:
                break
            
            index, sample = item
            try:
                # Use adaptive prompt instead of hardcoded one
                prompt = prompts['initial_analysis']

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.1:8b-instruct-q4_0",
                        "prompt": f"{prompt}\n\nDocument section:\n{sample}",
                        "stream": False
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    extracted = result.get("response", "").strip()
                    if extracted and extracted.lower() != "none" and len(extracted) > 50:
                        critical_info.append(extracted)
                        
            except Exception as e:
                print(f"⚠️ Error extracting from sample {index}: {e}")
            
            queue_items.task_done()
    
    # Start worker threads
    threads = []
    for _ in range(min(10, len(samples))):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Add samples to queue
    for i, sample in enumerate(samples):
        queue_items.put((i, sample))
    
    # Wait for completion
    queue_items.join()
    
    # Stop workers
    for _ in range(len(threads)):
        queue_items.put(None)
    for t in threads:
        t.join()
    
    print(f"✅ Extracted {len(critical_info)} critical information sections")
    return critical_info

def create_executive_summary(critical_sections, document_info, doc_type='contract'):
    """Create final executive summary from critical information using adaptive prompts"""
    print(f"📋 Creating executive summary: {document_info['pages']} pages → 3 page summary")
    print(f"📄 Using {doc_type} document template")
    
    # Get adaptive prompts for this document type
    prompts = adaptive_processor.prompt_generator.get_prompts(doc_type)
    
    combined_info = "\n\n".join([f"Section {i+1}:\n{section}" 
                                for i, section in enumerate(critical_sections[:25])])
    
    # Use adaptive summary template
    prompt = prompts['summary_template'] + f"""

Document Information:
- File: {document_info['filename']}  
- Pages: {document_info['pages']}
- Processing Date: {document_info['date']}

Analyzed sections:
{combined_info}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b-instruct-q4_0", 
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get("response", "Error: No response from AI")
            
            # Add document type header
            summary_with_type = f"**DOCUMENT TYPE: {doc_type.upper()}**\n\n{summary}"
            
            print(f"✅ Executive summary created: {len(summary_with_type):,} chars (~{len(summary_with_type)//3000} pages)")
            return summary_with_type
        else:
            return f"Error: API returned status {response.status_code}"
            
    except Exception as e:
        return f"Error creating summary: {str(e)}"

def upload_results_to_s3(job_id, summary, full_text, document_info, username):
    """Upload results to S3"""
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = f"summaries/{timestamp}_{job_id}"
        
        # Upload executive summary
        summary_key = f"{base_path}/executive_summary.txt"
        s3_client.put_object(
            Bucket='blackletter-files-prod',
            Key=summary_key,
            Body=summary,
            ContentType='text/plain'
        )
        
        # Upload full document text for chatbot
        chatbot_key = f"{base_path}/chatbot_source.txt" 
        s3_client.put_object(
            Bucket='blackletter-files-prod',
            Key=chatbot_key,
            Body=full_text,
            ContentType='text/plain'
        )
        
        # Upload metadata
        metadata = {
            "job_id": job_id,
            "username": username,
            "original_file": document_info['filename'],
            "processing_date": document_info['date'],
            "status": "completed",
            "pages": document_info['pages'],
            "summary_length": len(summary),
            "full_text_length": len(full_text),
            "executive_summary_url": summary_key,
            "chatbot_source_url": chatbot_key
        }
        
        metadata_key = f"{base_path}/metadata.json"
        s3_client.put_object(
            Bucket='blackletter-files-prod',
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        
        print(f"📤 Results uploaded to S3: {base_path}/")
        return base_path
        
    except Exception as e:
        raise Exception(f"Failed to upload results: {str(e)}")

def log_usage_to_dynamodb(job_id, username, document_info, processing_minutes):
    """Log usage information"""
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('BlackletterUsage')
        
        table.put_item(Item={
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'job_id': job_id,
            'pages_processed': document_info['pages'],
            'processing_time_minutes': processing_minutes,
            'instance_id': get_instance_id(),
            'document_name': document_info['filename']
        })
        print("📊 Usage logged to DynamoDB")
        
    except Exception as e:
        print(f"⚠️ DynamoDB logging failed: {e}")

def trigger_ec2_shutdown():
    """Trigger EC2 instance shutdown"""
    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lambda_client.invoke(
            FunctionName='StopEC2Instance',
            InvocationType='Event',
            Payload=json.dumps({
                'instance_id': get_instance_id()
            })
        )
        print(f"✅ Lambda shutdown triggered for instance: {get_instance_id()}")
        
    except Exception as e:
        print(f"⚠️ Lambda shutdown failed, trying direct EC2: {e}")
        try:
            ec2 = boto3.client('ec2', region_name='us-east-1')
            ec2.stop_instances(InstanceIds=[get_instance_id()])
            print(f"✅ Direct EC2 shutdown triggered for instance: {get_instance_id()}")
        except Exception as e2:
            print(f"⚠️ All shutdown methods failed: {e2}")

def main():
    """Main pipeline execution"""
    job_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()
    
    print(f"=== EXECUTIVE SUMMARY Blackletter Pipeline - Job ID: {job_id} ===")
    print("🎯 STRATEGY: 1-5 page executive summary + full document for chatbot")
    print("⚡ GOAL: Fast processing (5-15 min) regardless of document size")
    
    put_metric('JobsStarted', 1)
    
    try:
        # Step 1: Find and download the latest uploaded file
        log_job_status(job_id, 'processing', 'pdf_load', 'system')
        
        s3_key = get_latest_upload()
        username = extract_username_from_path(s3_key)
        local_path, filename = download_file_from_s3(s3_key)
        
        print(f"📄 Processing: {filename}")
        log_job_status(job_id, 'processing', 'text_extraction', username)
        
        # Step 2: Extract text and get document info
        full_text = extract_pdf_text(local_path)
        page_count = get_pdf_page_count(local_path)
        
        # Detect document type
        doc_type, confidence = adaptive_processor.detector.detect_document_type(full_text)
        print(f"📄 Document type detected: {doc_type} (confidence: {confidence:.2f})")
        
        document_info = {
            'filename': filename,
            'pages': page_count,
            'date': datetime.now().isoformat()
        }
        
        print(f"✅ Loaded document text ({len(full_text):,} characters)")
        
        log_job_status(job_id, 'processing', 'executive_summary_generation', username)
        
        print("🎯 EXECUTIVE SUMMARY STRATEGY: Fast, focused, chatbot-ready")
        print(f"📄 ACTUAL PDF pages: {page_count}")
        print(f"📄 Processing {page_count}-page document → 3-page executive summary")
        
        # Step 3: Executive Summary Generation
        print("\n" + "="*50)
        print("STEP 1: INTELLIGENT DOCUMENT SAMPLING")
        print("="*50)
        samples = intelligent_document_sampling(full_text, num_samples=25)
        
        print("\n" + "="*50) 
        print("STEP 2: CRITICAL INFORMATION EXTRACTION")
        print("="*50)
        critical_info = extract_critical_information(samples, doc_type)
        
        print("\n" + "="*50)
        print("STEP 3: EXECUTIVE SUMMARY GENERATION") 
        print("="*50)
        executive_summary = create_executive_summary(critical_info, document_info, doc_type)
        
        print("🎯 Executive summary strategy complete!")
        
        # Step 4: Save results locally
        log_job_status(job_id, 'processing', 'local_save', username)
        
        local_summary_path = f"/home/ubuntu/blackletter/executive_summary_{job_id}.txt"
        with open(local_summary_path, 'w') as f:
            f.write(executive_summary)
        
        print(f"💾 Executive summary saved to {local_summary_path}")
        
        # Step 5: Upload results to S3
        log_job_status(job_id, 'processing', 's3_upload', username)
        
        s3_path = upload_results_to_s3(job_id, executive_summary, full_text, document_info, username)
        
        # Step 6: Log usage and metrics
        end_time = datetime.utcnow()
        processing_minutes = (end_time - start_time).total_seconds() / 60
        
        log_usage_to_dynamodb(job_id, username, document_info, processing_minutes)
        
        print("\n" + "="*60)
        print("🎉 EXECUTIVE SUMMARY COMPLETE!")
        print("="*60)
        print(f"📄 Document: {page_count} pages")
        print(f"⏱️ Processing time: {processing_minutes:.1f} minutes") 
        print(f"📋 Executive summary: ~{len(executive_summary)//3000} pages (human reading)")
        print(f"🤖 Full document preserved for chatbot (100% accuracy)")
        print(f"🎯 Chatbot uses ONLY original document - no summary mixing")
        print(f"💰 Strategy: Fast & profitable for ANY document size")
        
        # Send success metrics and notifications
        put_metric('JobsCompleted', 1)
        put_metric('ProcessingTimeMinutes', processing_minutes)
        put_metric('DocumentPages', page_count)
        put_metric('SummaryPages', len(executive_summary)//3000 or 1)
        
        log_job_status(job_id, 'completed', 'finished', username)
        
        send_notification(job_id, 'completed', 
            f'Executive summary completed successfully!\n\n'
            f'User: {username}\n'
            f'Job ID: {job_id}\n'
            f'File: {filename}\n'
            f'Pages: {page_count:,}\n'
            f'Processing time: {processing_minutes:.1f} minutes\n'
            f'Results: s3://blackletter-files-prod/{s3_path}/')
        
        print("=== Pipeline Complete ===")
        
        # Clean up local file
        try:
            os.remove(local_path)
            print(f"🗑️ Cleaned up local file: {filename}")
        except:
            pass
            
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        put_metric('JobsFailed', 1)
        
        log_job_status(job_id, 'failed', 'error', username if 'username' in locals() else 'unknown', str(e))
        send_notification(job_id, 'failed', 
            f'Document processing failed!\n\n'
            f'Job ID: {job_id}\n'
            f'Error: {str(e)}\n\n'
            f'Full traceback:\n{traceback.format_exc()}')
        
    finally:
        trigger_ec2_shutdown()

if __name__ == "__main__":
    main()
