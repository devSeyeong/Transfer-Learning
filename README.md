# 전이 학습(Transfer Learning) 및 파인 튜닝(Fine Tuning) 프로젝트

이 프로젝트는 **InceptionV3** 사전 학습된 모델을 사용하여 이진 분류 문제(강아지와 고양이 분류)를 해결하는 예제입니다. 전이 학습을 통해 사전 학습된 가중치를 활용하고, 파인 튜닝을 통해 모델의 특정 레이어를 재학습하여 성능을 향상시키는 방법을 학습합니다.

---

## 프로젝트 개요

1. **전이 학습(Transfer Learning)**:
    - **InceptionV3** 모델의 사전 학습된 가중치를 가져와 새로운 데이터셋에 적용합니다.
    - 모델의 대부분의 레이어를 고정(freeze)하고, 최종 레이어에 새로운 분류 레이어를 추가합니다.
2. **파인 튜닝(Fine Tuning)**:
    - 사전 학습된 모델의 일부 레이어를 재학습하여 새로운 데이터셋에 맞도록 모델 성능을 더욱 개선합니다.

---

## 실행 방법

### 1. 데이터 준비 및 다운로드

- **Kaggle Dogs vs Cats Redux** 데이터셋을 다운로드합니다.

```python
python
코드 복사
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/'
!kaggle competitions download -c dogs-vs-cats-redux-kernels-edition

```

- 데이터셋을 적절히 전처리하고 `train_ds`와 `val_ds` 데이터셋으로 준비합니다.

---

### 2. 사전 학습된 모델 가중치 다운로드

- **InceptionV3**의 사전 학습된 가중치를 다운로드합니다.

```python
python
코드 복사
import requests

url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
r = requests.get(url, allow_redirects=True)
open('inception_v3.h5', 'wb').write(r.content)

```

---

### 3. 전이 학습(Transfer Learning)

- **InceptionV3** 모델을 로드하고, 사전 학습된 가중치를 불러옵니다.

```python
python
코드 복사
from keras.applications import InceptionV3

inception_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
inception_model.load_weights('inception_v3.h5')
inception_model.summary()

```

- 모델의 기존 레이어를 고정(freeze)하여 학습되지 않도록 설정합니다.

```python
python
코드 복사
for i in inception_model.layers:
    i.trainable = False

```

- 최종 레이어를 추가하여 새롭게 분류 문제를 해결하도록 설정합니다.

```python
python
코드 복사
import tensorflow as tf

마지막레이어 = inception_model.get_layer('mixed7')

layer1 = tf.keras.layers.Flatten()(마지막레이어.output)
layer2 = tf.keras.layers.Dense(1024, activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(drop1)

model = tf.keras.Model(inception_model.input, layer3)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_ds, validation_data=val_ds, epochs=2)

```

---

### 4. 파인 튜닝(Fine Tuning)

- 특정 레이어(`mixed6` 이후)를 학습 가능하도록 설정합니다.

```python
python
코드 복사
unfreeze = False
for i in inception_model.layers:
    if i.name == 'mixed6':
        unfreeze = True
    if unfreeze:
        i.trainable = True

```

- 모델을 다시 컴파일하고, 낮은 학습률로 재학습합니다.

```python
python
코드 복사
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc'])
model.fit(train_ds, validation_data=val_ds, epochs=2)

```

---

## 파일 구조

```bash
bash
코드 복사
프로젝트 폴더/
│
├── inception_v3.h5                     # InceptionV3 가중치 파일
├── train.py                            # 훈련 스크립트
├── 데이터셋/                            # Kaggle 데이터셋 디렉토리
│   ├── train/                          # 훈련 데이터
│   └── validation/                     # 검증 데이터
└── README.md                           # 프로젝트 설명 파일

```

---

## 주요 기능

1. **전이 학습**:
    - 사전 학습된 InceptionV3 모델의 강력한 특성 추출 기능을 활용하여 이진 분류 문제를 해결합니다.
2. **파인 튜닝**:
    - 모델의 특정 레이어를 재학습하여 성능을 더욱 향상시킵니다.
    - 낮은 학습률을 설정하여 이미 학습된 가중치가 크게 변경되지 않도록 조정합니다.

---

## 필수 라이브러리

이 프로젝트를 실행하기 위해 다음의 라이브러리가 필요합니다:

```bash
bash
코드 복사
pip install tensorflow keras requests kaggle

```

---

## 참고 사항

1. **데이터 준비**:
    - 데이터는 반드시 `(150, 150, 3)` 크기의 RGB 이미지로 전처리되어야 합니다.
2. **학습률 조정**:
    - 파인 튜닝 시 낮은 학습률(lr=0.00001)을 사용해야 모델의 기존 학습된 가중치가 크게 변경되지 않습니다.
3. **GPU 활용**:
    - 전이 학습 및 파인 튜닝은 계산량이 많으므로, GPU 환경에서 실행하는 것을 권장합니다.
