import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 주기와 족 뽑기 함수
def 파생_특징(z):
    족 = ((z - 1) % 18) + 1
    주기 = ((z - 1) // 18) + 1
    return 족, 주기

# 2. 데이터 읽기 및 파생 변수 추가 함수
def load_data(filename, nrows=None):
    데이터 = pd.read_csv(filename, nrows=nrows)
    데이터["족"], 데이터["주기"] = zip(*데이터["atomic_number"].apply(파생_특징))
    return 데이터

#학습 데이터(원자번호 및 원자 반지름)
읽을값 = 40

# 11. 학습 설정
학습률 = 0.001
반복 = 500000


# 3. 학습용 데이터 일부만 읽기 (예: 50개)
학습데이터 = load_data("elements.csv", nrows = 읽을값 )
X_train_raw = 학습데이터[["atomic_number", "족", "주기"]].values.astype(np.float32)
y_train = 학습데이터["atomic_radius"].values.astype(np.float32).reshape(-1,1)

# 4. 전체 데이터 불러오기 (118개)
전체데이터 = load_data("elements.csv")
X_all_raw = 전체데이터[["atomic_number", "족", "주기"]].values.astype(np.float32)
y_all = 전체데이터["atomic_radius"].values.astype(np.float32).reshape(-1,1)

# 5. 입력값 표준화 (학습 데이터 기준 fit)
scaler = StandardScaler()
입력_학습 = scaler.fit_transform(X_train_raw)
입력_전체 = scaler.transform(X_all_raw)

N = 입력_학습.shape[0]

# 6. 신경망 초기화 (He 초기화)
np.random.seed(0)
숨은_뉴런1 = 8
숨은_뉴런2 = 8

W1 = np.random.randn(3, 숨은_뉴런1) * np.sqrt(2/3)
b1 = np.zeros((1, 숨은_뉴런1))

W2 = np.random.randn(숨은_뉴런1, 숨은_뉴런2) * np.sqrt(2/숨은_뉴런1)
b2 = np.zeros((1, 숨은_뉴런2))

W3 = np.random.randn(숨은_뉴런2, 1) * np.sqrt(2/숨은_뉴런2)
b3 = np.zeros((1,1))

# 7. 활성화 함수 및 미분
def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)

def leaky_relu_derivative(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx

# 8. 순전파
def forward(X):
    z1 = X @ W1 + b1
    a1 = leaky_relu(z1)
    z2 = a1 @ W2 + b2
    a2 = leaky_relu(z2)
    z3 = a2 @ W3 + b3
    return z1, a1, z2, a2, z3

# 9. 손실함수 MSE
def mse(y, yhat):
    return np.mean((y - yhat)**2)

# 10. 역전파
def backward(X, y, z1, a1, z2, a2, yhat):
    m = len(X)

    dz3 = (yhat - y) / m
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0, keepdims=True)

    da2 = dz3 @ W3.T
    dz2 = da2 * leaky_relu_derivative(z2)

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * leaky_relu_derivative(z1)

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# 12. 학습 루프
for step in range(반복):
    z1, a1, z2, a2, yhat = forward(입력_학습)
    loss = mse(y_train, yhat)
    dW1, db1, dW2, db2, dW3, db3 = backward(입력_학습, y_train, z1, a1, z2, a2, yhat)

    W1 -= 학습률 * dW1
    b1 -= 학습률 * db1
    W2 -= 학습률 * dW2
    b2 -= 학습률 * db2
    W3 -= 학습률 * dW3
    b3 -= 학습률 * db3

    if step % 500 == 0:
        print(f"{step}회 손실(MSE): {loss:.4f}, dW1 norm: {np.linalg.norm(dW1):.6f}")

# 13. 전체 데이터에 대해 예측
_, _, _, _, 예측전체 = forward(입력_전체)

# 14. R² 계산 함수
def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot

print("전체 데이터 R² :", r2_score(y_all, 예측전체))

# 15. 예측값 일부 출력
print("원자번호 10 → 예측 반지름:", 예측전체[9,0])  # 10은 인덱스 9
print("원자번호 11 → 예측 반지름:", 예측전체[10,0])
print("원자번호 12 → 예측 반지름:", 예측전체[11,0])
print("원자번호 118 → 예측 반지름:", 예측전체[117,0])

# 16. 시각화 (전체 데이터 기준)
plt.figure(figsize=(10,6))
plt.plot(전체데이터["atomic_number"], y_all.flatten(), label=f"real (train size = {읽을값})", marker='o', linestyle='-', color='blue')
plt.plot(전체데이터["atomic_number"], 예측전체.flatten(), label="predicted", marker='x', linestyle='--', color='red')
plt.xlabel("(Atomic Number)")
plt.ylabel("(Atomic Radius)")
plt.title("real vs predicted atomic radius")
plt.legend()
plt.grid(True)
plt.show()
