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

# Configuration
MAX_WORKERS = min(20, os.cpu_count() * 2)  # Scale with CPU cores
CHUNK_TIMEOUT = 180  # 3 minutes per chunk
SUMMARY_TIMEOUT = 300  # 5 minutes for final summary
MAX_RETRIES = 3  # Retry failed chunks
CHUNK_SIZE = 8000  # Smaller chunks for Q&A

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

def chunk_document(text, chunk_size=CHUNK_SIZE):
    """Split document text into chunks for processing"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size // 4):  # Step by 1/4 chunk for overlap
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    
    print(f"📦 Created {len(chunks):,} chunks")
    return chunks

def upload_results_to_s3(job_id, summary, full_text, document_info, username):
    """Upload results to S3"""
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = f"summaries/{username}/{timestamp}_{job_id}"
        
        # Upload executive summary
        summary_key = f"{base_path}/executive_summary.txt"
        s3_client.put_object(
            Bucket='blackletter-files-prod',
            Key=summary_key,
            Body=summary,
            ContentType='text/plain',
            Metadata={
                'job_id': job_id,
                'username': username,
                'filename': document_info['filename'],
                'pages': str(document_info['pages'])
            }
        )
        
        # Upload full text for chatbot
        chatbot_key = f"{base_path}/chatbot_source.txt"
        s3_client.put_object(
            Bucket='blackletter-files-prod',
            Key=chatbot_key,
            Body=full_text,
            ContentType='text/plain'
        )
        
        # Create metadata file
        metadata = {
            "job_id": job_id,
            "username": username,
            "filename": document_info['filename'],
            "processing_date": document_info['date'],
            "pages": document_info['pages'],
            "summary_length": len(summary),
            "full_text_length": len(full_text),
            "executive_summary_url": f"s3://blackletter-files-prod/{summary_key}",
            "chatbot_source_url": f"s3://blackletter-files-prod/{chatbot_key}"
        }
        
        metadata_key = f"{base_path}/metadata.json"
        s3_client.put_object(
            Bucket='blackletter-files-prod',
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        
        print(f"📤 Uploaded results to: s3://blackletter-files-prod/{base_path}/")
        return base_path
        
    except Exception as e:
        raise Exception(f"Failed to upload results: {str(e)}")

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
            Subject=f'Blackletter Processing {status.upper()}: {job_id}'
        )
        print(f"📧 Notification sent: {status}")
        
    except Exception as e:
        print(f"⚠️ Failed to send notification: {str(e)}")

def put_metric(metric_name, value, unit='Count'):
    """Put CloudWatch metric"""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
        
        cloudwatch.put_metric_data(
            Namespace='Blackletter/Processing',
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
    
    print("✅ Environment validated")

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
    """Main processing pipeline"""
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
        
        # Create chunks
        log_job_status(job_id, 'processing', 'chunking', username)
        chunks = chunk_document(full_text)
        
        # Generate Q&A ready document
        log_job_status(job_id, 'processing', 'qa_processing', username)
        
        print(f"\n{'='*50}")
        print("📋 Q&A DOCUMENT PREPARATION")
        print(f"{'='*50}")
        
        print(f"✅ Document ready for Q&A with {len(chunks)} chunks")
        
        # Create simple Q&A ready marker
        qa_document = f"""**BLACKLETTER AI - DOCUMENT READY FOR Q&A**
**Date Processed: {datetime.now().strftime('%B %d, %Y')}**
**Original Document: {document_info['filename']} ({document_info['pages']} pages)**

This document has been processed into {len(chunks)} chunks and is ready for Q&A.
Ask any specific questions about the content to get detailed answers.
"""
        
        # Save locally
        local_summary_path = f"/home/ubuntu/blackletter/qa_ready_{job_id}.txt"
        with open(local_summary_path, 'w') as f:
            f.write(qa_document)
        print(f"💾 Q&A document saved: {local_summary_path}")
        
        # Upload to S3
        log_job_status(job_id, 'processing', 'upload', username)
        s3_path = upload_results_to_s3(job_id, qa_document, full_text, document_info, username)
        
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
        # trigger_ec2_shutdown()
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ PROCESSING FAILED: {error_msg}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        
        log_job_status(job_id, 'failed', 'error', 'unknown', error_msg)
        send_notification(job_id, 'failed', f'❌ Processing failed: {error_msg}')
        
        # Still shutdown on failure
        #  trigger_ec2_shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()
