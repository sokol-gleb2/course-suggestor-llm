from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from ..utils.config import DATA_PATHS
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def preprocess(row):
    courses = "\n".join(row["course_list"])
    return {
        "text": f"Job: {row['job']}\nCourses:\n{courses}",
        "label": int(row["label"])
    }

def tokenise(example):
    return tokeniser(example["text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, pred.predictions[:, 1])
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc
    }



if __name__ == "__main__":
    courseList_job_judge_scores = pd.read_json(DATA_PATHS["courseList_job_judge_scores"]) 
    data = courseList_job_judge_scores.apply(preprocess, axis=1).tolist()
    dataset = Dataset.from_list(data)

    # Tokenise
    tokeniser = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenised_dataset = dataset.map(tokenise)

    # Split
    split = tokenised_dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Define Trainer
    training_args = TrainingArguments(
        output_dir="./distilbert-job-eval",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokeniser,
        compute_metrics=compute_metrics # for eval
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model("./distilbert-job-evaluator")

   
    # Evaluating
    metrics = trainer.evaluate()
    print(metrics)
    
    # If I want to deploy
    # huggingface-cli login
    # from transformers import push_to_hub
    # model.push_to_hub("gleb/distilbert-job-evaluator")
    # tokeniser.push_to_hub("gleb/distilbert-job-evaluator")
    
    # Then later:
    # model = DistilBertForSequenceClassification.from_pretrained("gleb/distilbert-job-evaluator")

