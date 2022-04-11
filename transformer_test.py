# @File:transformer_test.py
# @Author:Fan
# @Time:2022/1/17 11:09


from transformers import pipeline

# # Allocate a pipeline for sentiment-analysis
# classifier = pipeline('sentiment-analysis')
# classifier('We are very happy to introduce pipeline to the transformers repository.')
# # [{'label': 'POSITIVE', 'score': 0.9996980428695679}]


from transformers import pipeline

# Allocate a pipeline for question-answering
# question_answerer = pipeline('question-answering')
# question_answerer({
#      'question': 'What is the name of the repository ?',
#      'context': 'Pipeline has been included in the huggingface/transformers repository'
#  })
# {'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}


# from transformers import AutoTokenizer, AutoModel
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")
#
# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
#
#
# import torch
# from transformers import BertModel, BertTokenizer
# # 这里我们调用bert-base模型，同时模型的词典经过小写处理
# model_name = 'bert-base-uncased'
# # 读取模型对应的tokenizer
# tokenizer = BertTokenizer.from_pretrained(model_name)
# # 载入模型
# model = BertModel.from_pretrained(model_name)
# # 输入文本
# input_text = "Here is some text to encode"
# # 通过tokenizer把文本变成 token_id
# input_ids = tokenizer.encode(input_text, add_special_tokens=True)
# # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
# input_ids = torch.tensor([input_ids])
# # 获得BERT模型最后一个隐层结果
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples


from transformers import pipeline

ner_pipe = pipeline("ner")

sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO,
therefore very close to the Manhattan Bridge which is visible from the window."""

result = ner_pipe(sequence)

for entity in result:
    print(entity)


