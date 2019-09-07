# README

kpx(전력거래소) 공모전을 준비하며 작성한 코드를 간단하게 정리함.

# get-data

**kma_forecast_scrapper**

- 기상청 기상예보 크롤링

**weatherpoing_api**

- sk웨더퐁 데이터 api 불러오기

# preprocess

#### preprocess_kma_forecast

- 기상청 날씨누리 기상예보 데이터 preprocess
- data/raw/kma/ 안에 데이터 넣기

#### preprocess_kma_observation

- 기상청 날씨누리 기상 관측치 데이터 preprocess
- data/raw/kma/ 안에 데이터 넣기

#### preprocess_kospo_power_data

- 남부발전 발전량 데이터 preprocess
- data/raw/kospo/wind/ 안에 데이터 넣기

#### preprocess_kpx_power_data

- kpx 발전량 데이터 preprocess
- data/raw/kpx/wind/ 안에 데이터 넣기



# eda

**kpx의  preprocess를 eda폴더안에 preprocess로 symbolic link를 걸어야 함**

- data/ 안에 데이터 넣기

각 지역별로

#### 예보(eda_weather_forecast)

- 발전량 데이터 경향성 살펴보기
- 풍향 인코딩용 polar_historgram확인
- 변수간 correlation
- 변수간 normalized mutual information
- 변수별 발전량과의 relative density plot
  - 변수별로 구간별 발전량 평균 보기
- 변수별 distribution plot
- Windowing
- 제주 타지역 발전량 데이터 경향성 살펴보기
- 특정 시간에 나온 예보 결과만 확인해보기

#### 관측치(eda_weather_observation)

- 1분 단위 관측치 데이터를 1시간, 3시간, 하루 평균으로 살펴봄
  - 변수간 correlation
  - 변수간 normalized mutual information
  - 변수별 발전량과의 relative density plot
  - 변수별 distribution plot



# models

**kpx의  preprocess와 eda를  models 폴더안에 각각 preprocess, eda로 symbolic link를 걸어야 함**

data/ 에 데이터 넣기

#### make_fe_data

train / test data 만들기

- 풍향 12방위
- 풍향/풍속 인코딩
- 발전량 moving average
  - 전년도, 전체 평균
- feature windowing
- add time feature (hour, day, dayofyear, month, year)

#### model_train

data/에 데이터 넣기

- make_fe_data에서 만든 데이터를 그대로 불러와서 모델 돌리기

X : date, 발전량, 제외 변수

y : 발전량

- Stacking model은 model 폴더안에 있음
- Stacking을 Walk Forward로 검증
- 코드 주석처리한 부분
  - y, yhat plot찍기 -> figure 폴더에 저장
  - mse plot
  - moving average plot
  - feature importance
  - 실험 결과 csv저장 -> result 폴더에 저장

- nested cross validation
  - XGBoost

#### clustering

- kmeans
- 결과 PCA, T-SNE로 Plot 확인
- 차원축소 후 kmeans
- kmeans 후 차원축소

#### Linear_Regression

data/에 데이터 넣기

- y : 발전량 데이터
- X : 날짜, 발전량 등을 제외한 feature들로 구성된 데이터
- 발전량/풍속 , 발전량/모든 변수에 대해 linear regression
- train, test 각 날짜별 y, yhat plot

#### dataset_manager

- MLP, RNN dataloader

#### MLP

data/에 데이터 넣기

- 데이터 불러온 후  날짜, location등 float가 아닌 값을 제외하고 scaling을 해준 후에 다시 날짜 등을 붙여서 dataset manager(**dataset_manager.py**)로 처리
- 반드시 na값은 제거하고 돌려야 함

- class MLP에서 architecture 수정

#### RNN

data/에 데이터 넣기

- MLP와 똑같은 방식으로 데이터 na제거 후 , Scaling
- 기존 1시간 단위 데이터 timedelta(days=1)을 이용하여 하루 단위 데이터로 묶기
  - 한 sequence가 24시간으로 데이터 구성

#### conditional_VAE

https://github.com/timbmg/VAE-CVAE-MNIST 참고

**make_cvae_data**

data/ 에 데이터 넣기

- kpx, hk, ss 각 location이 다른 데이터들을 불러와서 location을 0,1,2로 인코딩 (데이터셋 개수만큼 location 인코딩)
- 3 데이터를 concat
- date,datetime,발전량 데이터를 columns에서 제거한 후 data scaling 
- location을 추가한 후 pkl로 저장

**vae_model**

cvae dataset

- location이 condition (y)이 되고, 나머지가 x가 됨

idx2onehot

- condition을 onehot encoding으로

VAE

- conditional option을 주면 cvae

**main**

- make_cvae_data에서 만들었던 데이터 불러오기
- conditional : contional vae여부 (default 1)
- encoder/decoder_layer_sizes : list형태로 layer size 입력
- num_condition -> 데이터셋의 갯수
- python main.py로 실행