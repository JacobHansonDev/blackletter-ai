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

def get_instance_id():
    return "i-016606398a9831c5a"

def log_job_status(job_id, status, stage, error_message=None):
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
            'username': 'jacob',
            'instance_id': get_instance_id()
        })
        print(f"📊 Status logged: {status} - {stage}")
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

def chunk_text(text):
    return [text[i:i+2000] for i in range(0, len(text), 2000)]  # Optimized chunk size

def batch_summarize_chunks(chunks, batch_size=10):
    """Process multiple chunks in parallel for much faster processing"""
    print(f"🚀 Starting batch processing of {len(chunks)} chunks with {batch_size} workers...")
    
    results = [None] * len(chunks)
    result_queue = queue.Queue()
    
    def worker():
        while True:
            item = result_queue.get()
            if item is None:
                break
            index, chunk = item
            try:
                prompt = "Summarize in exactly 50 words or less, focusing only on essential legal facts, key dates, and critical obligations:"
                full_prompt = f"{prompt}\n\n{chunk}"
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.1:8b-instruct-q4_0",
                        "prompt": full_prompt,
                        "stream": False
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results[index] = result.get("response", "Error: No response from AI")
                else:
                    results[index] = f"Error: API returned status {response.status_code}"
                    
            except Exception as e:
                results[index] = f"Error calling AI: {str(e)}"
            
            # Progress indicator
            completed = sum(1 for r in results if r is not None)
            if completed % 50 == 0:
                print(f"🔄 Progress: {completed}/{len(chunks)} chunks completed")
            
            result_queue.task_done()
    
    # Start worker threads
    num_workers = min(batch_size, len(chunks))
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Add chunks to queue
    for i, chunk in enumerate(chunks):
        result_queue.put((i, chunk))
    
    # Wait for completion
    result_queue.join()
    
    # Stop workers
    for _ in range(num_workers):
        result_queue.put(None)
    for t in threads:
        t.join()
    
    print(f"✅ Batch processing completed: {len(chunks)} chunks processed")
    return results

def summarize_chunk(chunk):
    """Single chunk processing (fallback)"""
    try:
        prompt = "Summarize this legal document section clearly and professionally, focusing on key facts, dates, parties, and obligations:"
        full_prompt = f"{prompt}\n\n{chunk}"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b-instruct-q4_0",
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Error: No response from AI")
        else:
            return f"Error: API returned status {response.status_code}"
            
    except Exception as e:
        return f"Error calling AI: {str(e)}"

def main():
    job_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()
    
    print(f"=== Blackletter Pipeline Start - Job ID: {job_id} ===")
    put_metric('JobsStarted', 1)
    
    try:
        log_job_status(job_id, 'processing', 'pdf_load')
        
        pdf_files = [f for f in os.listdir('/home/ubuntu/') if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in /home/ubuntu/")
            
        pdf_files.sort(key=lambda x: os.path.getmtime(f'/home/ubuntu/{x}'), reverse=True)
        latest_pdf = pdf_files[0]
        input_path = f"/home/ubuntu/{latest_pdf}"
        
        print(f"📄 Processing: {latest_pdf}")
        print(f"✅ Loaded {input_path} at {datetime.utcnow()}")
        
        log_job_status(job_id, 'processing', 'text_extraction')
        
        try:
            with open('/home/ubuntu/extracted_text.txt', 'r', encoding='utf-8') as f:
                raw_text = f.read()
            print(f"✅ Using extracted PDF text ({len(raw_text)} characters)")
        except Exception as e:
            print(f"⚠️ Could not read extracted text: {e}")
            raw_text = f"Processing real PDF: {latest_pdf}. " + "This is placeholder content for testing. " * 20
        
        log_job_status(job_id, 'processing', 'chunking')
        chunks = chunk_text(raw_text)
        print(f"📝 Chunked into {len(chunks)} parts.")
        
        log_job_status(job_id, 'processing', 'summarization')
        print("🤖 Starting batch summarization...")
        
        try:
            # Use batch processing for much faster results
            summaries = batch_summarize_chunks(chunks, batch_size=30)
            print("🤖 Batch summarization complete.")
        except Exception as e:
            print(f"⚠️ Batch processing failed, falling back to sequential: {str(e)}")
            summaries = [summarize_chunk(chunk) for chunk in chunks]
            print("🤖 Sequential summarization complete.")
        
        log_job_status(job_id, 'processing', 'local_save')
        output_path = f"/home/ubuntu/blackletter/output_{job_id}.txt"
        
        try:
            with open(output_path, "w") as f:
                f.write(f"Blackletter Summary for: {latest_pdf}\n")
                f.write("="*50 + "\n\n")
                for i, s in enumerate(summaries, 1):
                    f.write(f"Chunk {i}:\n{s}\n\n")
            print(f"💾 Output saved to {output_path}")
        except Exception as e:
            raise Exception(f"Failed to save output: {str(e)}")
        
        log_job_status(job_id, 'processing', 's3_upload')
        
        max_retries = 3
        s3_path = None
        
        for attempt in range(max_retries):
            try:
                s3_client = boto3.client('s3', region_name='us-east-1')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                s3_key = f"summaries/{timestamp}_{job_id}/summary.txt"
                
                s3_client.upload_file(output_path, 'blackletter-files-prod', s3_key)
                
                metadata = {
                    "job_id": job_id,
                    "original_file": latest_pdf,
                    "processing_date": datetime.now().isoformat(),
                    "status": "completed",
                    "summary_url": s3_key,
                    "chunks_processed": len(chunks)
                }
                
                s3_client.put_object(
                    Bucket='blackletter-files-prod',
                    Key=f'summaries/{timestamp}_{job_id}/metadata.json',
                    Body=json.dumps(metadata, indent=2),
                    ContentType='application/json'
                )
                
                s3_path = f"summaries/{timestamp}_{job_id}/"
                print(f"📤 Results uploaded to S3: {s3_path}")
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"S3 upload failed after {max_retries} attempts: {str(e)}")
                print(f"⚠️ S3 upload attempt {attempt + 1} failed, retrying...")
        
        try:
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            table = dynamodb.Table('BlackletterUsage')
            table.put_item(Item={
                'username': 'jacob',
                'timestamp': datetime.now().isoformat(),
                'pages_processed': len(chunks),
                'instance_id': get_instance_id(),
                'job_id': job_id
            })
            print("📊 Usage logged to DynamoDB")
        except Exception as e:
            print(f"⚠️ DynamoDB logging failed: {e}")
        
        end_time = datetime.utcnow()
        processing_minutes = (end_time - start_time).total_seconds() / 60
        
        put_metric('JobsCompleted', 1)
        put_metric('ProcessingTimeMinutes', processing_minutes)
        
        log_job_status(job_id, 'completed', 'finished')
        send_notification(job_id, 'completed', f'Document processing completed successfully!\n\nJob ID: {job_id}\nFile: {latest_pdf}\nProcessing time: {processing_minutes:.1f} minutes\nResults location: {s3_path}\nChunks processed: {len(chunks)}')
        
        print("=== Pipeline Complete ===")
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        put_metric('JobsFailed', 1)
        
        log_job_status(job_id, 'failed', 'error', str(e))
        send_notification(job_id, 'failed', f'Document processing failed!\n\nJob ID: {job_id}\nError: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}')
        
    finally:
        try:
            print(f"🔌 EC2 shutdown triggered for instance: {get_instance_id()}")
            ec2 = boto3.client('ec2', region_name='us-east-1')
            ec2.stop_instances(InstanceIds=[get_instance_id()])
        except Exception as e:
            print(f"⚠️ Failed to shutdown EC2: {e}")

if __name__ == "__main__":
    main()
