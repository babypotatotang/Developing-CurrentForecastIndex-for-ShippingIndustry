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
### Analysis Sentimental
(1) Crawling and Merging 
파일 위치: Developing-CurrentForecastIndex-for-ShippingIndustry/1. Analysis Sentimental/(1) Crawling & Merging/   
+ 네이버, 다음에서 다음의 데이터를 크롤링함. 
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
(2) Modeling    
+ 수집한 뉴스데이터 전처리
    + 조사, 어미 등의 단어를 모두 제거  
+ 
### Handling Shipping News
### Calculating Index
### Data Analysis 
 
**Environment** 

