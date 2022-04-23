library(forecast)
library(nowcasting)
library(Metrics)

#Read Index Data
IndexData<-readxl::read_excel(path="D:\\KMI-Project_tmp\\Data\\3. Shipping News\\18. Real Index.xlsx",sheet='Sheet1',col_types='numeric') #파일 열었음(data frame)

#전체 기간에 대해서 정상성 확인: BPanel의 Trans code를 3으로 설정했을 때 pvalue가 0.1로 모두 정상성을 만족함(전년대비 증가율)
# library(tseries)
# Data.Trans<-ts(IndexData[1:260,3:29],start=c(2001,1),frequency = 12) # 뉴스데이터가 없는 데이터셋(PI가 증감률이어서 2001.01 부터 시작)
# Transformation<-rep(3,26) #PI 값 제외하고
# 
# Trans.base<-Bpanel(base=Data.Trans[,-PI_position],trans=Transformation,h=110)
# 
# for (s in colnames(Trans.base)){
#   print(s)
#   print(Trans.base[1:250,s] %>% adf.test())
# }

#PI Value Trans 
y<-ts(IndexData[1:262,3:4],start=c(2000,1),frequency=12)
y<-Bpanel(y,trans = rep(3,2))
y<-ts(y[1:262,1],start=c(2000,1),frequency=12)

#News Index Trans
News<-ts(IndexData[1:262,29:30],start=c(2000,1),frequency=12)
News<-Bpanel(News,trans = rep(3,2))
News<-ts(News[1:262,2],start=c(2000,1),frequency=12)


#(1) 자기회귀모형
Model.AR<-arima(y,c(12,0,0)) #(p,d,q)에서 (AR order, degree of differencing, MA order)

Forecast.AR<-forecast::forecast(Model.AR,h=1)
Forecast.AR$fitted
y.AR<-ts(Forecast.AR$fitted,start=c(2000,1),frequency=12)# the predicted valus <- y h$fitat

#(2) 동태요인모형
ts<-ts(IndexData[1:262,4:29],start=c(2000,1),frequency = 12)
base<-Bpanel(ts,trans=rep(3,26))

data<-cbind(y,base)
colnames(data)<-c("Y",colnames(base))

#고유치 비율 확인 -> 성분 개수 설정 
Data<-base[1:262,]

PCA.Data<-princomp(Data,cor=T) #주성분분석
PCA.EigenValue<-PCA.Data$sdev^2 #성분의 고유치 연산 

EigenRatioV<-c() #고유치 비율 벡터 저장 
for(i in (1:14)){ #rmax: 7, kmax=2*rmax
  EigenValue1=PCA.EigenValue[i]
  EigenValue2=PCA.EigenValue[i+1]
  EigenRatio<-EigenValue1/EigenValue2
  EigenRatioV<-append(EigenRatioV,EigenRatio)
}

EigenRatio.max<-max(EigenRatioV)
EigenRatioV[which(EigenRatioV==EigenRatio.max)]

EigenRatioV

dfm<-nowcast(formula=Y~. ,data=data,method='2s',r=2,p=12,q=2,frequency=rep(12,27))
fm<-dfm$yfcst[1:262,1]

y.NoNews<-y.AR+fm

#뉴스데이터 지수 항 
auto.arima(News)

news.ar<-arima(News,c(0,0,1),seasonal = list(order=c(0,0,1)))
news.ar<-forecast(news.ar,h=1)
news.ar<-news.ar$fitted

res<-y-y.NoNews
data<-data.frame(res,news.ar)

modellm<-lm(formula=res~.,data)
news<-modellm$fitted.values

y.News<-y.NoNews+news

##검증 
rmse(y,y.AR) #0.06660604
rmse(y,y.NoNews) #0.03762398
rmse(y,y.News) #0.03754859

mae(y,y.AR) #0.04841153515725412
mae(y,y.NoNews) #0.02793947
mae(y,y.News) #0.02779436

dd<-data.frame(y,y.AR,y.NoNews,y.News)
write.csv(dd,'index_test.csv')

plot(y,type='l',main='자기회귀 모형 예측력 비교', xlab='TimeSeries',ylab='Production Index')+lines(y.AR,type='l',col='blue')
plot(y,type='l',main='동태요인 모형 예측력 비교', xlab='TimeSeries',ylab='Production Index')+lines(y.NoNews,type='l',col='blue')
plot(y,type='l',main='당기예측 모형 예측력 비교', xlab='TimeSeries',ylab='Production Index')+lines(y.News,type='l',col='blue')
