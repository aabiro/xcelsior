#!/usr/bin/env python3
"""
Xcelsior Sample Inference Script
Demonstrates GPU usage with BERT model for text classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def check_gpu():
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU available: {gpu_count} device(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({memory_total:.2f} GB)")
        return True
    else:
        print("✗ No GPU available. Running on CPU.")
        return False

def run_inference():
    """Run BERT inference on sample text."""
    print("\n" + "="*60)
    print("BERT Sentiment Analysis Demo")
    print("="*60)
    
    # Check GPU
    has_gpu = check_gpu()
    device = "cuda" if has_gpu else "cpu"
    
    # Load model and tokenizer
    print("\nLoading BERT model...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    print(f"✓ Model loaded on {device}")
    
    # Sample texts
    texts = [
        "Xcelsior is an amazing GPU scheduler!",
        "I'm having a terrible day.",
        "The weather is okay, I guess.",
        "This is the best project I've ever worked on!",
        "I'm disappointed with the results."
    ]
    
    print("\n" + "-"*60)
    print("Running inference on sample texts:")
    print("-"*60)
    
    # Run inference
    for i, text in enumerate(texts, 1):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = "POSITIVE" if predictions[0][1] > predictions[0][0] else "NEGATIVE"
            confidence = predictions[0][1 if sentiment == "POSITIVE" else 0].item()
        
        # Print result
        print(f"\n{i}. Text: \"{text}\"")
        print(f"   Sentiment: {sentiment} (confidence: {confidence:.2%})")
    
    # GPU memory usage
    if has_gpu:
        memory_used = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        print("\n" + "-"*60)
        print(f"GPU Memory Used: {memory_used:.2f} GB")
        print(f"GPU Memory Cached: {memory_cached:.2f} GB")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)

if __name__ == "__main__":
    run_inference()
