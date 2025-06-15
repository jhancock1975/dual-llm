#!/usr/bin/env python
import boto3
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualModelChatbot:
    def __init__(self, region_name='us-east-1'):
        """
        Initialize the dual model chatbot with AWS Bedrock.
        
        Args:
            region_name: AWS region for Bedrock service
        """
        self.bedrock = boto3.client(
            service_name='bedrock',
            region_name=region_name
        )
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        
        # Model configuration
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        # Model states
        self.model_a_frozen = True  # Start with model A frozen
        self.model_b_frozen = False
        
        # Training data buffer
        self.training_buffer = []
        self.swap_probability = 0.2  # 20% chance to swap models after each interaction
        
        # Fine-tuning job tracking
        self.active_fine_tuning_jobs = {
            'model_a': None,
            'model_b': None
        }
        
    def invoke_model(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Invoke the frozen model to generate a completion.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in the response
            
        Returns:
            Generated completion text
        """
        # Prepare the request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            completion = response_body['content'][0]['text']
            
            return completion
            
        except Exception as e:
            logger.error(f"Error invoking model: {str(e)}")
            return ""
    
    def prepare_training_data(self, prompt: str, completion: str) -> Dict:
        """
        Prepare training data in the format required for fine-tuning.
        
        Args:
            prompt: Original prompt
            completion: Generated completion
            
        Returns:
            Formatted training example
        """
        # Concatenate prompt and completion as specified
        combined_input = f"{prompt}\n\nAssistant: {completion}"
        
        training_example = {
            "messages": [
                {
                    "role": "user",
                    "content": combined_input
                },
                {
                    "role": "assistant",
                    "content": completion
                }
            ]
        }
        
        return training_example
    
    def save_training_data(self, training_examples: List[Dict], s3_bucket: str, s3_key: str):
        """
        Save training data to S3 in JSONL format for fine-tuning.
        
        Args:
            training_examples: List of training examples
            s3_bucket: S3 bucket name
            s3_key: S3 object key
        """
        s3_client = boto3.client('s3')
        
        # Convert to JSONL format
        jsonl_content = '\n'.join([json.dumps(example) for example in training_examples])
        
        try:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=jsonl_content.encode('utf-8')
            )
            logger.info(f"Training data saved to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Error saving training data: {str(e)}")
    
    def start_fine_tuning(self, model_name: str, training_data_s3_uri: str, output_s3_uri: str):
        """
        Start a fine-tuning job for the specified model.
        
        Args:
            model_name: 'model_a' or 'model_b'
            training_data_s3_uri: S3 URI for training data
            output_s3_uri: S3 URI for output model
        """
        job_name = f"chatbot-fine-tune-{model_name}-{int(time.time())}"
        
        try:
            response = self.bedrock.create_model_customization_job(
                jobName=job_name,
                customModelName=f"chatbot-{model_name}-{int(time.time())}",
                roleArn="arn:aws:iam::071350569379:role/BedrockFineTuningRole",  # Replace with your role
                baseModelIdentifier=self.model_id,
                trainingDataConfig={
                    "s3Uri": training_data_s3_uri
                },
                outputDataConfig={
                    "s3Uri": output_s3_uri
                },
                hyperParameters={
                    "epochCount": "1",
                    "batchSize": "1",
                    "learningRate": "0.00001"
                }
            )
            
            self.active_fine_tuning_jobs[model_name] = response['jobArn']
            logger.info(f"Started fine-tuning job: {job_name}")
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning job: {str(e)}")
    
    def maybe_swap_models(self):
        """
        Randomly swap which model is frozen and which is being fine-tuned.
        """
        if random.random() < self.swap_probability:
            logger.info("Swapping models...")
            self.model_a_frozen = not self.model_a_frozen
            self.model_b_frozen = not self.model_b_frozen
            
            # Clear training buffer after swap
            self.training_buffer = []
            
            logger.info(f"Model A frozen: {self.model_a_frozen}, Model B frozen: {self.model_b_frozen}")
    
    def chat(self, user_input: str, s3_bucket: str, s3_prefix: str) -> str:
        """
        Process a chat interaction with the dual model system.
        
        Args:
            user_input: User's input message
            s3_bucket: S3 bucket for storing training data
            s3_prefix: S3 prefix for organizing training data
            
        Returns:
            Model's response
        """
        # Generate completion using the frozen model
        completion = self.invoke_model(user_input)
        
        # Prepare training data
        training_example = self.prepare_training_data(user_input, completion)
        self.training_buffer.append(training_example)
        
        # Save training data periodically (every 10 examples)
        if len(self.training_buffer) >= 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{s3_prefix}/training_data_{timestamp}.jsonl"
            
            self.save_training_data(self.training_buffer, s3_bucket, s3_key)
            
            # Start fine-tuning on the non-frozen model
            if self.model_a_frozen and not self.model_b_frozen:
                output_uri = f"s3://{s3_bucket}/{s3_prefix}/models/model_b/"
                self.start_fine_tuning(
                    "model_b",
                    f"s3://{s3_bucket}/{s3_key}",
                    output_uri
                )
            elif self.model_b_frozen and not self.model_a_frozen:
                output_uri = f"s3://{s3_bucket}/{s3_prefix}/models/model_a/"
                self.start_fine_tuning(
                    "model_a",
                    f"s3://{s3_bucket}/{s3_key}",
                    output_uri
                )
            
            # Clear buffer after starting fine-tuning
            self.training_buffer = []
            
            # Maybe swap models only after fine-tuning starts
            self.maybe_swap_models()
        
        return completion
    
    def get_fine_tuning_status(self, model_name: str) -> Dict:
        """
        Check the status of a fine-tuning job.
        
        Args:
            model_name: 'model_a' or 'model_b'
            
        Returns:
            Job status information
        """
        job_arn = self.active_fine_tuning_jobs.get(model_name)
        if not job_arn:
            return {"status": "No active job"}
        
        try:
            response = self.bedrock.get_model_customization_job(
                jobIdentifier=job_arn
            )
            return {
                "status": response['status'],
                "jobName": response['jobName']
            }
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return {"status": "Error", "error": str(e)}


# Example usage
def main():
    # Initialize the chatbot
    chatbot = DualModelChatbot(region_name='us-east-1')
    
    # Configuration
    S3_BUCKET = "dual-llm"  # Replace with your S3 bucket
    S3_PREFIX = "chatbot-training"
    
    print("Dual Model Chatbot initialized. Type 'quit' to exit.")
    print(f"Model A frozen: {chatbot.model_a_frozen}")
    print(f"Model B frozen: {chatbot.model_b_frozen}")
    print()
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'status':
            print("\nFine-tuning Status:")
            print(f"Model A: {chatbot.get_fine_tuning_status('model_a')}")
            print(f"Model B: {chatbot.get_fine_tuning_status('model_b')}")
            print(f"Model A frozen: {chatbot.model_a_frozen}")
            print(f"Model B frozen: {chatbot.model_b_frozen}")
            print(f"Training buffer size: {len(chatbot.training_buffer)}")
            print()
            continue
        
        # Get response from chatbot
        response = chatbot.chat(user_input, S3_BUCKET, S3_PREFIX)
        print(f"Assistant: {response}")
        print()


if __name__ == "__main__":
    main()