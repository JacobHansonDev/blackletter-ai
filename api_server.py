#!/usr/bin/env python3
"""
Blackletter AI API Server
Serves processed documents to the frontend
"""
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from datetime import datetime
import os

app = FastAPI(title="Blackletter AI API", version="1.0.0")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS clients (with region specified)
s3_client = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# Configuration
RESULTS_BUCKET = 'blackletter-files-prod'
DYNAMODB_TABLE = 'blackletter-usage'

@app.get("/documents")
def list_documents():
    """List all processed documents"""
    try:
        # List objects in results bucket
        response = s3_client.list_objects_v2(
            Bucket=RESULTS_BUCKET,
            MaxKeys=100  # Limit for performance
        )
        
        documents = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Filter out non-document files
                if obj['Key'].endswith(('.txt', '.pdf')):
                    documents.append({
                        'id': obj['Key'].replace('.txt', '').replace('.pdf', ''),
                        'name': obj['Key'],
                        'size': obj['Size'],
                        'lastModified': obj['LastModified'].isoformat(),
                        'url': f"https://{RESULTS_BUCKET}.s3.amazonaws.com/{obj['Key']}"
                    })
        
        # Sort by last modified (newest first)
        documents.sort(key=lambda x: x['lastModified'], reverse=True)
        
        return {
            'documents': documents,
            'count': len(documents),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'documents': [],
            'count': 0
        }

@app.get("/documents/{document_id}")
def get_document(document_id: str):
    """Get a specific document's content"""
    try:
        # Try to get the text summary first
        key = f"{document_id}.txt"
        
        response = s3_client.get_object(Bucket=RESULTS_BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')
        
        return {
            'id': document_id,
            'content': content,
            'type': 'summary',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found: {str(e)}")

@app.post("/documents/upload")
def upload_document():
    """Placeholder for document upload - currently handled via S3 directly"""
    return {
        'message': 'Document upload currently handled via S3 bucket upload',
        'bucket': 'blackletter-input',
        'instructions': 'Upload PDF files directly to S3 bucket for processing'
    }

@app.get("/status")
def get_processing_status():
    """Get current processing status"""
    try:
        # Check if any processing is currently running
        # This is a simplified check - you might want to enhance this
        return {
            'status': 'ready',
            'message': 'System ready for document processing',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'Blackletter AI API',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }

if __name__ == "__main__":
    print("Starting Blackletter AI API server...")
    print("Server will be available at: http://0.0.0.0:8002")
    print("Health check: http://0.0.0.0:8002/health")
    print("Documents endpoint: http://0.0.0.0:8002/documents")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002,
        reload=False,
        access_log=True
    )