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
CHUNK_SIZE = 50000  # Characters per chunk (~20 pages)

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
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"📦 Downloading {s3_key} to {local_path} (attempt {attempt + 1})")
            s3_client.download_file('blackletter-files-prod', s3_key, local_path)
            
            file_size = os.path.getsize(local_path)
            print(f"✅ Downloaded {filename} ({file_size:,} bytes)")
            return local_path, filename
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ Download failed, retrying... {e}")
                time.sleep(2)
            else:
                raise

def extract_username_from_path(s3_key):
    """Extract username from S3 path"""
    try:
        parts = s3_key.split('/')
        if len(parts) >= 3 and parts[0] == 'uploads':
            username = parts[1]
            if '@' in username or username != 'unknown_user':
                return username
        return 'unknown_user'
    except:
        return 'unknown_user'

def extract_pdf_text(pdf_path):
    """Extract text from PDF with progress tracking"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            text_parts = []
            
            print(f"📄 Extracting text from {total_pages} pages...")
            
            # Process in batches for memory efficiency
            batch_size = 50
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                batch_text = []
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text:
                            batch_text.append(page_text)
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                        continue
                
                text_parts.extend(batch_text)
                print(f"🔄 Processed pages {batch_start + 1}-{batch_end}/{total_pages}")
            
            full_text = '\n'.join(text_parts)
            char_count = len(full_text)
            
            if char_count < 100:
                raise Exception(f"Extracted text too short ({char_count} chars). PDF might be scanned/image-based.")
            
            print(f"✅ Extracted {char_count:,} characters from {total_pages} pages")
            return full_text, total_pages
            
    except Exception as e:
        raise Exception(f"Failed to extract PDF text: {str(e)}")

def chunk_document(text, chunk_size=CHUNK_SIZE):
    """Simple chunking - no fancy sampling"""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"📄 Document split into {len(chunks)} chunks")
    return chunks

def summarize_chunk(chunk, chunk_index):
    """Summarize one chunk - super simple"""
    prompt = f"Summarize this section of a document. Focus on the key information:\n\n{chunk}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b-instruct-q4_0",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9
                    }
                },
                timeout=CHUNK_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                if summary and len(summary) > 50:
                    return summary
                elif attempt < MAX_RETRIES - 1:
                    print(f"⚠️ Chunk {chunk_index} summary too short, retrying...")
                    time.sleep(1)
            else:
                print(f"⚠️ Chunk {chunk_index} API error: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error processing chunk {chunk_index}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    
    return None

def summarize_chunks_parallel(chunks):
    """Summarize all chunks in parallel"""
    print(f"🔍 Summarizing {len(chunks)} chunks in parallel...")
    print(f"⚙️ Using {MAX_WORKERS} workers")
    
    summaries = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(summarize_chunk, chunk, i): i 
            for i, chunk in enumerate(chunks)
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result:
                    summaries[index] = result
                    completed += 1
                    if completed % 5 == 0:
                        print(f"🔄 Completed {completed}/{len(chunks)} chunks")
            except Exception as e:
                print(f"⚠️ Failed to process chunk {index}: {e}")
    
    # Return summaries in order
    ordered_summaries = []
    for i in range(len(chunks)):
        if i in summaries:
            ordered_summaries.append(summaries[i])
    
    print(f"✅ Successfully summarized {len(ordered_summaries)} chunks")
    return ordered_summaries

def create_executive_summary(chunk_summaries, document_info):
    """Create final summary from chunk summaries"""
    print(f"📋 Creating executive summary from {len(chunk_summaries)} chunk summaries")
    
    combined = "\n\n".join([
        f"Section {i+1} Summary:\n{summary}" 
        for i, summary in enumerate(chunk_summaries)
    ])
    
    prompt = f"""Create a comprehensive executive summary of this document based on the section summaries below.

Write a professional 3-5 page executive summary that includes:
- What this document is about (overview and purpose)
- Who the key parties or stakeholders are
- The main points, provisions, or findings
- Important dates, deadlines, and timelines
- Financial amounts or obligations
- Key requirements or actions needed
- Critical takeaways and implications

Make it clear, well-organized, and useful for decision-making.

Section summaries to synthesize:
{combined}"""

    try:
        print("🤖 Generating final executive summary...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b-instruct-q4_0",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "seed": 42,
                    "num_predict": 4000  # Allow for 3-5 page summary
                }
            },
            timeout=SUMMARY_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get("response", "Error: No response from AI")
            
            # Clean up any meta-commentary
            lines_to_remove = ["Based on", "I'll create", "Let me", "Here is", "I will"]
            summary_lines = summary.split('\n')
            cleaned_lines = [
                line for line in summary_lines 
                if not any(line.strip().startswith(phrase) for phrase in lines_to_remove)
            ]
            summary = '\n'.join(cleaned_lines).strip()
            
            # Add header
            final_summary = f"""**BLACKLETTER AI EXECUTIVE SUMMARY**
**Date Processed: {datetime.now().strftime('%B %d, %Y')}**
**Original Document: {document_info['filename']} ({document_info['pages']} pages)**

---

{summary}

---

*This executive summary was generated by Blackletter AI. For detailed analysis and specific questions about this document, please use the chatbot feature.*"""
            
            print(f"✅ Executive summary created: {len(final_summary):,} characters")
            return final_summary
            
        else:
            raise Exception(f"API returned status {response.status_code}")
            
    except Exception as e:
        raise Exception(f"Failed to create summary: {str(e)}")

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
                'original-pages': str(document_info['pages']),
                'job-id': job_id
            }
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
        
        print(f"📤 Results uploaded to S3: s3://blackletter-files-prod/{base_path}/")
        return base_path
        
    except Exception as e:
        raise Exception(f"Failed to upload results: {str(e)}")

def log_job_status(job_id, status, stage, username, error_message=None):
    """Log job status to DynamoDB"""
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        table = dynamodb.Table('BlackletterUsage')
        
        table.put_item(Item={
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'stage': stage,
            'username': username,
            'error_message': error_message or '',
            'instance_id': get_instance_id()
        })
        
        print(f"📊 Status logged: {status} - {stage}")
    except Exception as e:
        print(f"⚠️ Failed to log status: {e}")

def send_notification(job_id, status, message):
    """Send SNS notification"""
    try:
        sns = boto3.client('sns', region_name='us-east-1')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:577638390288:blackletter-notifications',
            Subject=f'Blackletter Job {status.title()}: {job_id}',
            Message=message
        )
        print(f"📧 Notification sent: {status}")
    except Exception as e:
        print(f"⚠️ Failed to send notification: {e}")

def put_metric(metric_name, value, unit='Count'):
    """Send CloudWatch metric"""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
        cloudwatch.put_metric_data(
            Namespace='Blackletter/Pipeline',
            MetricData=[{
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now()
            }]
        )
        print(f"📈 Metric sent: {metric_name} = {value}")
    except Exception as e:
        print(f"⚠️ Failed to send metric: {e}")

def validate_environment():
    """Validate Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama API not responding")
        
        models = response.json().get('models', [])
        if not any('llama3.1:8b' in m.get('name', '') for m in models):
            raise Exception("Required model llama3.1:8b not found")
        
        print("✅ Environment validated: Ollama is running with required model")
    except Exception as e:
        raise Exception(f"Environment validation failed: {str(e)}")

def trigger_ec2_shutdown():
    """Trigger EC2 shutdown"""
    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lambda_client.invoke(
            FunctionName='StopEC2Instance',
            InvocationType='Event',
            Payload=json.dumps({'instance_id': get_instance_id()})
        )
        print("✅ EC2 shutdown triggered")
    except Exception as e:
        print(f"⚠️ Failed to trigger shutdown: {e}")
        # Try direct shutdown as fallback
        try:
            ec2 = boto3.client('ec2', region_name='us-east-1')
            ec2.stop_instances(InstanceIds=[get_instance_id()])
            print("✅ Direct EC2 shutdown triggered")
        except:
            pass

def main():
    """Main pipeline execution"""
    job_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"🚀 BLACKLETTER AI PIPELINE - Job ID: {job_id}")
    print(f"{'='*70}")
    print(f"📅 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    put_metric('JobsStarted', 1)
    
    try:
        # Validate environment
        validate_environment()
        
        # Download document
        log_job_status(job_id, 'processing', 'download', 'system')
        s3_key = get_latest_upload()
        username = extract_username_from_path(s3_key)
        local_path, filename = download_file_from_s3(s3_key)
        
        print(f"\n📄 Processing: {filename}")
        print(f"👤 User: {username}")
        
        # Extract text
        log_job_status(job_id, 'processing', 'extraction', username)
        full_text, page_count = extract_pdf_text(local_path)
        
        document_info = {
            'filename': filename,
            'pages': page_count,
            'date': datetime.now().isoformat()
        }
        
        # Generate summary
        log_job_status(job_id, 'processing', 'summary', username)
        
        print(f"\n{'='*50}")
        print("📋 EXECUTIVE SUMMARY GENERATION")
        print(f"{'='*50}")
        
        # Chunk document
        chunks = chunk_document(full_text)
        
        # Summarize chunks in parallel
        chunk_summaries = summarize_chunks_parallel(chunks)
        
        if not chunk_summaries:
            raise Exception("No chunk summaries generated")
        
        # Create final summary
        executive_summary = create_executive_summary(chunk_summaries, document_info)
        
        # Save locally
        local_summary_path = f"/home/ubuntu/blackletter/executive_summary_{job_id}.txt"
        with open(local_summary_path, 'w') as f:
            f.write(executive_summary)
        print(f"💾 Summary saved: {local_summary_path}")
        
        # Upload to S3
        log_job_status(job_id, 'processing', 'upload', username)
        s3_path = upload_results_to_s3(job_id, executive_summary, full_text, document_info, username)
        
        # Log completion
        processing_time = (datetime.now() - start_time).total_seconds() / 60
        log_job_status(job_id, 'completed', 'finished', username)
        
        # Send metrics
        put_metric('JobsCompleted', 1)
        put_metric('ProcessingTimeMinutes', processing_time)
        put_metric('DocumentPages', page_count)
        
        # Success summary
        print(f"\n{'='*60}")
        print(f"🎉 PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📄 Document: {filename}")
        print(f"📖 Pages: {page_count:,}")
        print(f"⏱️  Time: {processing_time:.1f} minutes")
        print(f"💾 Location: s3://blackletter-files-prod/{s3_path}/")
        print(f"👤 User: {username}")
        print(f"{'='*60}")
        
        # Send notification
        send_notification(job_id, 'completed',
            f'✅ Executive summary completed!\n\n'
            f'Document: {filename}\n'
            f'Pages: {page_count:,}\n'
            f'Processing time: {processing_time:.1f} minutes\n'
            f'User: {username}\n\n'
            f'Results available in S3.')
        
        # Cleanup
        try:
            os.remove(local_path)
            print("🗑️  Cleaned up local file")
        except:
            pass
            
    except Exception as e:
        # Error handling
        error_msg = str(e)
        print(f"\n❌ PROCESSING FAILED: {error_msg}")
        
        log_job_status(
            job_id, 
            'failed',
            'error',
            username if 'username' in locals() else 'unknown',
            error_msg
        )
        
        put_metric('JobsFailed', 1)
        
        send_notification(job_id, 'failed',
            f'❌ Processing failed!\n\n'
            f'Job ID: {job_id}\n'
            f'Error: {error_msg}\n\n'
            f'Please check the logs.')
        
        raise
        
    finally:
        # Always shutdown EC2
        print("\n🔌 Initiating EC2 shutdown...")
        trigger_ec2_shutdown()
        print(f"⏱️  Total time: {(datetime.now() - start_time).total_seconds():.1f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        trigger_ec2_shutdown()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        trigger_ec2_shutdown()
        sys.exit(1)
