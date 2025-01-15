# News based maritime index
(2021~2022) News based maritime index    
_Update: 2022-04-28_     

## **Index**
+ [About this project](#about-this-project)   
+ [Overview](#overview)   
  + [Goal](#goal)   
  + [Flow](#flow)   
+ [Detail Function](#detail-function)
    + [Analysis Sentimental](#analysis-sentimental)
    + [Handling Shipping News](#handling-shipping-news)
    + [Calculating Index](#calculating-index)
    + [Data Analysis](#data-analysis)
+ [Environment](#environment)   


## **About this project**    
+ 프로젝트 이름: 뉴스데이터를 활용한 해운업 경기 당기 예측 지수 개발          
+ 프로젝트 진행 목적:  2021 공공빅데이터 인턴십 수련 활동   
    + 한국해양수산개발원(KMI): https://www.kmi.re.kr/      
+ 프로젝트 진행 기간:  2021년 9월 ~ 2022년 2월   
+ 프로젝트 참여 인원:  1명               

## **Overview** 
> ### **Goal**   
+ (목적) 뉴스데이터를 활용한 텍스트마이닝 기법과 감성분석을 통해 뉴스데이터 지수를 산출하고 경기 예측에 사용하기 위함.     
+ (필요성) 다소 생소한 해운업 분야의 실물경기 현 상황과 변화 방향을 신속하게 파악하는 것과 경제 주체들의 민첩한 대응책 마련하기 위함.   
> ### **Flow** 
<img src = "https://user-images.githubusercontent.com/68631435/164913390-64bea86e-8a7e-4820-bafa-6b00673a44bc.png" width="60%" height="40%"> 

## **Detail Function**    
### **Analysis Sentimental**        
> **(1) Crawling and Merging**     
파일 위치: Developing-CurrentForecastIndex-for-ShippingIndustry/1. Analysis Sentimental/(1) Crawling & Merging/        
+ 모델 학습 데이터 구축을 위해 네이버, 다음에서 다음의 데이터를 크롤링함.    
검색 뉴스: 많이 본 뉴스/ 댓글 많은 뉴스 (2021년 3월 ~ 2021년 11월)   
    + 뉴스 날짜
    + 뉴스 제목
    + 뉴스 본문
    + 뉴스 URL
    + 뉴스 아래 감성 rating 수치
        + 좋아요
        + 감동이에요
        + 슬퍼요
        + 화가 나요.  
+ 각 기사의 감성지수는 다음의 식을 통해 산출함.    
    + 긍정 rating(좋아요, 감동이에요) - 부정 rating (슬퍼요, 화가 나요)   
    + 양수이면 1(긍정) tag, 음수이면 0(부정) tag   
+ 크롤링 후 전체 기사 Merge, 월별로 Merge     
> **(2) Modeling**        
> **전처리** 
```python
def common_word_list(common_num,neg,pos):
    negative_word=[]; positive_word=[]
    n_list=neg.most_common(common_num); p_list=pos.most_common(common_num)

    for i in range(common_num):
        negative_word.append(n_list[i][0])
        positive_word.append(p_list[i][0])

    common_list=list(set(negative_word) & set(positive_word))

    print(common_list)
    print('common_list 길이', len(common_list))

    return common_list

# #tokenized를 list로 변경
mecab=Mecab()
stopwords = ['했','있','으로','로','것','씨','말','도', '는', '다', '의', '가', '이', '은','수','에서','한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네',    '들', '듯', '지', '임', '게', '만', '겜', '되', '음', '면']

train_data['tokenized']=train_data['Sentence'].apply(mecab.morphs) #Sentence 내용을 morphs로 형태소 분석(type: list)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords]) #해당 열의 값 중 stopword에 해당하는 값 지우기
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if len(item)>1]) #길이 2이상만 저장
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in common_list]) #해당 열의 값 중 stopword에 해당하는 값 지우기
```
+ 다음의 기준으로 기사 본문 데이터에 대해 전처리를 진행함.    
    + (1) 조사, 어미 등으로 구성된 stopword 제거   
    + (2) 단어의 길이가 2보다 작은 경우 제거    
    + (3) common word list를 생성하고(함수 common_word_list), 그에 해당하는 단어 제거

>**정수 인코딩 및 패딩**
```python
### 정수 인코딩 ###
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train) #문자데이터를 입력받아 리스트 형태로 변환, 각 단어에 index 부여

vocab_size = total_cnt - rare_cnt + 2 # 사용되는 단어 집합의 크기

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') #새 vocab_size로 tokenizer 새로 설정
tokenizer.fit_on_texts(X_train)

#X_train, X_test의 데이터를 넣어서 인코딩 
X_train = tokenizer.texts_to_sequences(X_train)

### 패딩 ### 
def below_threshold_len(max_len, nested_list):
# 희귀 단어의 개수만큼 제거하는 함수, max_len은 리뷰의 최대 및 평균 길이를 보고 비교해서 설정
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 1000
below_threshold_len(max_len, X_train)
X_train = pad_sequences(X_train, maxlen = max_len)
```
+ 정수 인코딩 범위 설정함.    
    + 전체 단어의 개수(total cnt)와 임계치(threshold)보다 작은 경우에 해당하는 희귀 단어 수(rare cnt)를 계산    
+ 패딩    
    + max_len의 값을 임의로 변경하며 샘플 비율을 확인하고(함수 below_threshold_len), pad_sequence 실시    

>**모델 생성**
```python 
embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units)))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=256, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
```   
+ 다음 패키지를 설치하고 모델링 실시   
    + from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional   
    + from tensorflow.keras.models import Sequential, load_model   
    + from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   
+ Bidirectional-LSTM 방식으로 데이터를 학습함.    
+ 손실율: 0.4597 // 정확도: 0.8141     
### Handling Shipping News
> **(1) Crawling**     
파일 위치: Developing-CurrentForecastIndex-for-ShippingIndustry/2. Handling Shipping News/Crawling_뉴스데이터_Shipping.ipynb     
+ 감성 분류기 모델에 input으로 들어갈 데이터 구축을 위해 bigkinds 사이트에서 뉴스데이터를 크롤링함.   
+ 크롤링한 데이터는 다음과 같음.    
    + 검색 키워드: 해운업,해운산업,해운경기,해운업계    
    + 검색 기간: 2000년 1월 ~ 2021년 11월   
    + 뉴스 제목   
    + 뉴스 날짜   
    + 뉴스 본문    
    + 뉴스 url  


> **(2) Topic Modeling**       
**각 80개 토픽의 상위 25개 연관어를 추출 후 정합성 검증 후 NMF 토픽을 사용하였음.**    
>**LDA Topic Modeling**
```   python 
# 설치 패키지
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_texts
from gensim.test.utils import datapath

# common_texts에서 dictionary 생성
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# corpus를 활용하여 LdaModel 생성
lda = LdaModel(common_corpus, num_topics=80)

#document(뉴스데이터)에서 word 추출 (말뭉치 생성)
data_word=[[word for word in x.split(' ')] for x in document] 
id2word=corpora.Dictionary(data_word)

texts=data_word
corpus=[id2word.doc2bow(text) for text in texts]

print("Corpus Ready")

#생성한 말뭉치로 lda 시작 
lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=80)
print("lda done, please wait")

#출력부
for i in range(num_topics):
    words = model.show_topic(i, topn=num_words); #반환하는 토픽 연관어 개수 
    word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]

print("Result_out")
```   
+ gensim 패키지 활용하여 LDA Topic Modeling
+ 80개 토픽으로 나누어 분류

> **NMF Topic Modeling**
``` python 
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

#Count Vector 생성
vectorizer=CountVectorizer(analyzer='word')
x_counts=vectorizer.fit_transform(text)

transformer=TfidfTransformer(smooth_idf=False)
x_tfidf=transformer.fit_transform(x_counts)

xtfidf_norm=normalize(x_tfidf,norm='l2',axis=1)

print("xtfidf_norm Ready")

model=NMF(n_components=80,init='nndsvd')
model.fit(xtfidf_norm) # xtidf 데이터를 fit함

print("Model Ready")

for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    
    print(f'For topic {topic+1} the words with the highest value are:')
    
    print(tmp.nlargest(25))
#출력부
```
+ sklearn을 활용하여 NMF Topic Modeling     
+ 마찬가지로 80개 토픽을 분류   
### Calculating Index   
> **(1) Topic Count**    
+ 2000년 1월부터 2021년 10월까지의 뉴스데이터에서 NMF 방식으로 추출한 각 토픽별 연관어의 개수 집계    
+ 월별 지수 산출을 위해 각 뉴스데이터의 일별 토픽 단어 수를 집계함     
> **(2) Sentimental Index** 
>**Daily Sentimental**
```python
# 감성지수를 분석하는 함수 
def sentiment_predict(new_sentence):
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩

    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    return score
```
+ 위 함수를 통해 각 뉴스데이터의 긍정, 부정 지수를 predict함. 
+ 예측 후에는 단순 감성지수에 해당하는 긍정-부정 값을 덧붙여주었음. 
> **Monthly Sentimental**
``` python 
for i in LCount: #월별 뉴스 개수 
    index=LCount.index(i) 
    df_tmp=df_Sentimental[pre:pre+i]
    
    #일별 감성지수의 평균값을 감성지수
    pos=df_tmp['Pos'].tolist(); MeanPos=np.mean(pos); 
    neg=df_tmp['Neg'].tolist(); MeanNeg=np.mean(neg)
    
    SentiIndex=(MeanPos-MeanNeg)*100
    LSenti.append(round(SentiIndex,1))
    pre=i
```
+ 다음의 절차를 통해 일별로 예측한 감성지수를 월별 지수로 변환하였음. 
    + (1) 월별 뉴스 개수 만큼 긍정 수치와 부정수치의 평균을 구함. 
    + (2) (평균 긍정 - 평균 부정)*100 으로 감성지수를 산출
> **(3) Index**
+ 뉴스데이터지수 산출의 경우 선행연구를 따라 식을 설계하였음.       
    + 결합지수 1: 감성지수 * 토픽 비중 상위 20개 토픽의 10개 연관어 비중 (%)   
    + 결합지수 2: 감성지수 * 토픽 간 상관 상위 20개 토픽 단어 비중 (%)   
    + 결합지수 3: 감성지수*토픽-생산 상관 상위 20개 토픽 단어 비중 (%)   
        여기서 해운업 생산 상관성을 비교하기 위해 수상운송업생산지수를 참고하였음. (통계청)    
+ 3개의 지수와 실제지표 간 높은 상관성을 띄는 결합지수 3을 뉴스데이터 지수로 선정하였음. 
    + 실제 지표: OECD에서 발표한 우리나라의 산업생산지수        
    + 지수와 실제 지표 간 상관계수          
        |결합지수(1)|결합지수(2)|결합지수(3)|
        |:---:|:---:|:---:|
        |-0.295|-0.343|-0.493|
### Data Analysis 
> **(1) Data Set Ready**
+ 모형을 만들기 전 뉴스데이터 이외에 당기예측모형에 적용될 해운업 실물 데이터셋을 구축함. 
+ Stopford(2008), Chen et al(2015), Choi,Kim and Han(2018)을 참고하여 해운시장의 공급분야, 수요분야, 운임 및 가격 분야, 경제상황 분야로 나누어 수집, 우리나라 해운업 생산지수를 예측하는 것을 목표로 하기에 KOSPI, CLI(KOREA) 등을 추가하여 26개의 실물 지표를 선정함.    
+ 월별 시계열 데이터에 해당하므로 안정된 시계열성을 띄기 위해 Bpanel(library tseries)를 활용하여 안정화함.
    + trans code는 3으로 전년대비 증가율에 해당
> **(2) Modling**   
<img src = "https://user-images.githubusercontent.com/68631435/165331687-2fcb9390-921e-44bc-acaf-653fb4f3f63c.png" width="60%" height="40%">      


+ Domenico Giannone(2008)이 제안한 당기예측모형을 활용하여 해운업 경기 예측을 시도하였음.      
+ 예측력 평가를 위해 3개의 모델을 만들었음.    
    + 자기회귀모형
    + 동태요인모형 : 실제 지표만 사용
    + 당기예측모형 : 실제 지표 + 뉴스데이터 지수 사용   
+ 각 모형의 RMSE와 MAE 비교   
    |분류|자기회귀모형|동태요인모형|당기예측모형|
    |:---:|:---:|:---:|:---:|
    |RMSE|0.06661|0.03762|0.03754|
    |MAE|0.04841|0.02794|0.02784|
+ 비교 결과 당기예측모형(실제 지표 + 뉴스데이터 지수)의 성능이 가장 좋았음.    

## **Environment**   
+ Python (3.7.3)  
+ R (4,1.2)   
+ JupyterNotebook   
