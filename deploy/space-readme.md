---
title: Financial NER Demo
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Financial NER Demo

Paste or upload financial-document text (bank statements, tax forms) and see
named entities highlighted: people, organizations, locations, dollar amounts,
dates, account numbers, SSNs, and form numbers.

DistilBERT fine-tuned on [FiNER-ORD](https://huggingface.co/datasets/gtfintechlab/finer-ord)
(7 base entity types), then extended to 17 labels via real classifier-weight
transfer plus synthetic financial-document data.

Models: [financial-ner-v1](https://huggingface.co/kavyaagar/financial-ner-v1) (base, 7 labels),
[financial-ner-extended](https://huggingface.co/kavyaagar/financial-ner-extended) (17 labels, used by this demo).

Source: https://github.com/Kavya-Agar/financial-ner-project
