# BERT-fastapi

## 과제 목표
- huggingface transformers 기반 모델과 fastapi 라이브러리로 다음의 기능을 수행하는 단순한 API 웹서버를 구현

### 요건
1. GLUE task 중 CoLA 데이터를 활용해, 주어진 문장이 문법적으로 적합한지를 판별하는 API를 개발
2. 최소한 3개의 huggingface transformers에서 제공하는 Pretrained model에 GLUE CoLA데이터셋을 입력해 finetunning하고, 이 중 BEST모형을 선택
3. 선택된 모형을 fastapi라이브러리를 활용해 서비스하고, 서비스 포트는 8000번을 활용
4. /generate 엔드포인트는 JSON payload를 HTTP POST 방식으로 입력받아 JSON response를 반환
5. 모델 평가방식은 Fine tunning metric(Matthew's corr)

## 과제 진행

### 사전학습 모델 선정
- State-of-the-Art에서 Linguistic Acceptability on CoLA페이지<sup>[1](#footnote_1)</sup>
  - Rank가 높은 모델부터 huggingface transformers 기반 모델로 finetuning을 진행한 논문을 참고하여 BERT, roberta, albert를 선정
  - 추가 참고자료 검색 중 발견한 코드에서 BERT, distilbert, distilroberta로 진행된 것을 확인
  - BERT, distilbert, distilroberta, roberta, albert 총 5개 모델로 테스트를 수행

### TrainingArguments 조정
- google-research 깃헙<sup>[2](#footnote_2)</sup>과 state-of-the-art 자료 참고하여 batch_size는 64, optimizer는 adamW, learning_rate 는 논문마다 기준이 달라서 여러 가지 기준으로 테스트가 필요할 것으로 판단되어 이번 과제는 TrainingArguments의 초기값 5e-5<sup>[3](#footnote_3)</sup>
로 진행
- GLUE 데이터셋 테스트시 epochs는 2, 3, 4회로 finetuning하는 것을 BERT논문<sup>[4](#footnote_4)</sup> A.3 Fine-tuning Procedure 에서 추천하고 있어, 4회까지 finetuning 진행

### 모델 성능 평가
||bert-base-uncased|distilbert-base-uncased|distilroberta-base|roberta-base|albert-base-v2|
|:--:|:--:|:--:|:--:|:--:|:--:|
|epoch 1|0.547|0.482|0.451|0.529|0|
|epoch 2|0.599|0.509|0.520|0.578|0|
|epoch 3|0.592|0.520|0.588|0.565|0|
|epoch 4|0.579|0.525|0.561|0.596|0|

- albert-base-v2 모델의 평가가 제대로 이루어지지 않은 원인은 현재로써는 불명. 추가 보완이 필요함
- 테스트 과정에서 가장 성능이 좋은 모델은 bert-base-uncased 로 확인

### api
- main.py 파일이 있는 디렉토리에서 아래의 명령어를 실행
```
uvicorn main:app --reload
```

- 아래와 같이 모델을 실행
```python
import requests
import json

url = 'http://localhost:8000/generate'
params = {'passage' : 'I are a boy'}
headers = {'accept' : 'application/json',
           'Content-Type' : 'application/json'}
           
res = requests.post(url, data = json.dumps(params), headers = headers)
```

- api 서비스를 통해서 결과물을 얻음
```
{'prob' : 0.2570807635784149}
```

<hr>
<a name="footnote_1">1</a>: https://paperswithcode.com/sota/linguistic-acceptability-on-cola <br>
<a name="footnote_2">2</a>: https://github.com/google-research/google-research/blob/master/cola/main.py <br>
<a name="footnote_3">3</a>: https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments <br>
<a name="footnote_4">4</a>: Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina (11 October 2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
