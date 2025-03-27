def predict_job_outcome(job, course_list, model, tokenizer):
    input_text = f"Job: {job}\nCourses:\n" + "\n".join(course_list)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()

    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item()
    
    return {
        "label": prediction,  # 0 = jobless, 1 = job secured
        "confidence": confidence
    }


if __name__ == "__main__":
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch

    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained("./distilbert-job-evaluator")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert-job-evaluator")

    model.eval()

    # Example
    result = predict_job_outcome(
        job="Data Scientist at Spotify",
        course_list=["Statistics 101", "Intro to Python", "Data Visualization"],
        model=model,
        tokenizer=tokenizer
    )

    print(result)
    # Output: {'label': 1, 'confidence': 0.89}