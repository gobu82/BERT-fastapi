import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = torch.load("./bert_base_uncased_555.pt", map_location=torch.device('cpu'))
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.eval()


@torch.no_grad()
def predict(text):
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    output = quantized_model(**tokens)
    logits = output["logits"].detach()
    prediction = logits.softmax(dim=1)[0][1].item()
    return prediction