# Developing-CurrentForecastIndex-for-ShippingIndustry
(2021~2022) Developing Current Forcast Index of the Shipping Industry applying News Data   
_Update: 2022-04-24_   

## **Index**
+ [About this project](#about-this-project)   
+ [Overview](#overview)   
  + [Goal](#goal)   
  + [Flow](#flow)   
+ [Detail Function](#detail-function)
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

>## **Detil Function**    
### **Analysis Sentimental**        
**(1) Crawling and Merging**     
파일 위치: Developing-CurrentForecastIndex-for-ShippingIndustry/1. Analysis Sentimental/(1) Crawling & Merging/        
+ 네이버, 다음에서 다음의 데이터를 크롤링함.    
검색 키워드: 해운업, 해운산업, 해운경기, 해운업계 (중 1개 이상 포함)    
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
**(2) Modeling**        
> 전처리 
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

> 정수 인코딩 및 패딩 
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

> 모델 생성
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
### Calculating Index
### Data Analysis 
 
**Environment** 

