# Fine-tuning DistilBert on FiNER-139 Dataset for NER

## Project Overview
This project focuses on fine-tuning the DistilBert model, a lighter version of BERT, for Named Entity Recognition (NER) tasks using the FiNER-139 dataset. The objective is to achieve high precision, recall, and F1 scores in identifying named entities across various domains. This README provides details on the model architecture, dataset, training process, evaluation results, and instructions for inference, including converting the model to ONNX format for broader applicability.

## Model Details
- **Model Architecture:** DistilBert (distilbert-base-uncased)
- **Fine-tuning Details:** The model was fine-tuned specifically for the NER task, adjusting the top layer to classify a set of entities defined in the FiNER-139 dataset.
- **Parameters:** 
  - Top Labels Count: 4
  - Total Rows Processed: 100,000
  - Label All Tokens: True
  - Max Length: 200
  - Padding: Max Length
  - Truncation: True
  - Batch Size: 64

## Dataset
- **Source:** `nlpaueb/finer-139`
- **Size:** A subset of 100,000 records was used from the dataset for training.
- **Selected Labels (4 most frequent):**
  - `O`
  - `B-DebtInstrumentFaceAmount`
  - `I-DebtInstrumentFaceAmount`
  - `B-DebtInstrumentBasisSpreadOnVariableRate1`
  - `I-DebtInstrumentBasisSpreadOnVariableRate1`
  - `B-LesseeOperatingLeaseTermOfContract`
  - `I-LesseeOperatingLeaseTermOfContract`
  - `B-ContractWithCustomerLiability`
  - `I-ContractWithCustomerLiability`


## Training Process
The DistilBert model was trained using the specified subset of the FiNER-139 dataset. The training involved tokenizing the input text, aligning labels with tokens, and performing fine-tuning over 5 epochs. The learning rate was set to 2e-5 with a weight decay of 0.01 for regularization.

## Evaluation Results
The model was evaluated on a test set separate from the training data, yielding the following overall metrics:
- **Precision:** 0.811
- **Recall:** 0.893
- **F1 Score:** 0.850
- **Accuracy:** 0.999

### Evaluation Results per Class
In addition to overall metrics, the model's performance was assessed for each entity type. Below are the evaluation results per class, providing insight into the model's ability to accurately identify and classify various named entities:

| Entity Type                                   | Precision | Recall  | F1-Score | Support |
|-----------------------------------------------|-----------|---------|----------|---------|
| ContractWithCustomerLiability                 | 0.895     | 0.607   | 0.723    | 28      |
| DebtInstrumentBasisSpreadOnVariableRate1      | 0.866     | 0.969   | 0.914    | 452     |
| DebtInstrumentFaceAmount                      | 0.754     | 0.844   | 0.797    | 469     |
| LesseeOperatingLeaseTermOfContract            | 1.000     | 0.500   | 0.667    | 8       |


## Trained Model Availability

The trained DistilBert model fine-tuned on the FiNER-139 dataset for NER tasks is available on Hugging Face. You can access and download the model for direct use or further experimentation from the following link:

[DistilBert Base Uncased Fine-tuned for NER on FiNER-139 - Hugging Face Model Hub](https://huggingface.co/OlesB/distilbert-base-uncased-finetuned-ner_finer_139)

## Inference
To perform inference with the trained model, follow these steps:

### Loading the Model and Tokenizer
```python
from transformers import DistilBertForTokenClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

labels_file = 'labels_finer-139_top_rows_100000_top_labels_4'
dbert_model = DistilBertTrainer()
model_name = "OlesB/distilbert-base-uncased-finetuned-ner_finer_139"
dataset_name = "nlpaueb/finer-139"
rows_count = 10

max_length = 200
padding = 'max_length'
truncation = True

labels_path = hf_hub_download(repo_id=model_name, filename=labels_file)
labels_list = np.load(labels_path)

model = DistilBertForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = dbert_model.load_data(dataset_name, rows_count)

tokenized_inputs = tokenizer(
    dataset["tokens"],
    truncation=truncation,
    is_split_into_words=True,
    max_length=max_length,
    padding=padding,
    return_tensors="pt"
  )

outputs = model(**tokenized_inputs)
predicted_labels_ids = outputs.logits.argmax(axis=2)
predicted_labels = [
    [labels_list[token] for token in seq if token != -100]
    for seq in predicted_labels_ids
]
```

