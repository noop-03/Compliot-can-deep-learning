import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False #한글 폰트 깨짐 방지용
import numpy as np # 보드 만들때 쓰는 거 배열 만들어줌
import torch # 딥러닝용 이건 나도 잘 모름
import torch.nn as nn # 딥러닝용 22
import torch.optim as optim 
import matplotlib.pyplot as plt #이걸로 UI따는거임
from tqdm import tqdm #프로그레스 바 만들어줌

BOARD_SIZE = 8
NUM_MINES = 10

# 전체 보드 생성 (숫자/지뢰 포함) -1이 지로ㅣ, 
def generate_board(size=BOARD_SIZE, num_mines=NUM_MINES):
    board = np.zeros((size, size), dtype=int)
    mines = np.random.choice(size * size, num_mines, replace=False)
    for idx in mines:
        x, y = divmod(idx, size)
        board[x, y] = -1
    for x in range(size):
        for y in range(size):
            if board[x, y] == -1:
                continue
            board[x, y] = sum(
                board[i, j] == -1
                for i in range(max(0, x-1), min(size, x+2))
                for j in range(max(0, y-1), min(size, y+2))
            )
    return board

# 일부 셀만 공개된 입력 상태 생성 (갯수 고정 아님 각각 배열요소마다 0.2 확률로 은닉상태 해제
def mask_board(board, p=0.2):
    masked = np.full_like(board, -1)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] != -1 and np.random.rand() < p:
                masked[i, j] = board[i, j]
    return masked

# 학습 데이터 생성(개선) 1000개 만들어서 동서남북 + 앞뒤 뒤집어서 8배로 뻥튀기
def create_dataset(n=1000):
    X, Y = [], []
    for _ in tqdm(range(n)):
        full = generate_board()
        p = np.random.uniform(0.1, 0.5)
        obs = mask_board(full, p)
        X.append(obs)
        Y.append((full == -1).astype(np.float32))
    
    # 데이터 셋들을 NumPy 배열로 변환해줌 (2차원: [샘플번호, 행, 열]) (3차원 이었는데 바꾼거 2차원이 더 효율적
    X_np = np.array(X)  # shape: (n, 8, 8)
    Y_np = np.array(Y)  # shape: (n, 8, 8)
    
    # 데이터 증강 (너무 느리면 이거 제거하셈
    augmented_X, augmented_Y = [], []
    for x, y in zip(X_np, Y_np):
        for k in range(4):  # 0, 90, 180, 270도 회전
            rot_x = np.rot90(x, k, axes=(0, 1))  # 2D 배열 회전회오리
            rot_y = np.rot90(y, k, axes=(0, 1))
            augmented_X.append(rot_x)
            augmented_Y.append(rot_y)
            
            # 수평 반전
            flip_x = np.flip(rot_x, axis=1)  # axis=1은 열 방향 반전
            flip_y = np.flip(rot_y, axis=1)
            augmented_X.append(flip_x)
            augmented_Y.append(flip_y)
    
    # 최종적으로 PyTorch 텐서로 변환 (차원 추가: [샘플번호, 채널, 행, 열])
    X = torch.tensor(np.stack(augmented_X)).unsqueeze(1).float()  # shape: (n*8, 1, 8, 8)
    Y = torch.tensor(np.stack(augmented_Y)).unsqueeze(1).float()
    return X, Y

# CNN 모델 정의(개선)
class ImprovedMineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential( #얘가 압축시키고 
            nn.Conv2d(1, 64, 3, padding=1), #(입력 개수, 출력개수, 필터 수(3*3이니까 3이라 써진거), 입력 주변에 픽 셀 추가하는지)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential( #얘가 다시 압축 해제시킴
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # 보면 아까랑 다르게 출력이 입력보다 개수가 적음 = 필터 제거 + 해상도 증가
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# 학습 루프(개선) 에포치를 여기서 처리하는 거  batch size를 32로 지정하고 애포치를 30번 하니까 iteration이 30번인 거임 
def train(model, X, Y, epochs=20, lr=1e-3, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) #여기부터
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, factor=0.5) #여기까지는 학습률을 최적화해 주는 옵티마이저랑 학습 스케쥴러 작성인데
    # 옵티마이저란? https://xpmxf4.tistory.com/56 귀찮으니 여기 참고
    loss_fn = nn.BCELoss() #손실 함수
    
    best_loss = float('inf') 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in loader:
            opt.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        
        # 검증 세트 평가
        model.eval()
        with torch.no_grad():
            val_pred = model(X[:200])  # 처음 200개를 검증용으로 사용
            val_loss = loss_fn(val_pred, Y[:200])
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 최고 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')


#평가 메트릭스
def evaluate(model, X, Y, threshold=0.5):
    with torch.no_grad():
        pred = model(X)
        pred_binary = (pred > threshold).float()
        
        # 정확도 (전체 예측중 맞춘거)
        accuracy = (pred_binary == Y).float().mean()
        
        # 정밀도(지뢰라고 한거중 진짜 지뢰 였던거), 재현율(실제 지뢰중 모델이 맞춘거), F1 점수 = 2* (정밀도*재현율/정밀도+재현율) 얼마나 재현율과 정밀도 사이의 조화가 있는지 판별 1에 가까울수록 좋음
        true_pos = (pred_binary * Y).sum()
        false_pos = (pred_binary * (1-Y)).sum()
        false_neg = ((1-pred_binary) * Y).sum()
        
        precision = true_pos / (true_pos + false_pos + 1e-7)# 정밀도 계산
        recall = true_pos / (true_pos + false_neg + 1e-7)#재현율 계산
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7) #f1 계산
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # ROC AUC 계산 (ROC= 곡선임 x=안전칸을 지뢰칸으로 오탐한 비율 y= 실제 맞춘 지뢰 비율 AUC 곡선 아래 면적임 1에 가까울수록 정확도가 높은 모델
        from sklearn.metrics import roc_auc_score
        y_true = Y.flatten().numpy()
        y_score = pred.flatten().numpy()
        roc_auc = roc_auc_score(y_true, y_score)
        print(f"ROC AUC: {roc_auc:.4f}")


# 예측 결과 시각화
def visualize(model):
    model.eval()
    board = generate_board() # 실제 지뢰판 생성
    masked = mask_board(board) # 부분 관측된 보드
    input_tensor = torch.tensor(masked).unsqueeze(0).unsqueeze(0).float()
    # 2) 모델 예측
    with torch.no_grad():
        pred = model(input_tensor).squeeze().numpy() # 지뢰 확률 맵 (8×8)
    # 3) 시각화
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 관측된 보드
    axs[0].imshow(masked, cmap='gray', vmin=-1, vmax=8)
    axs[0].set_title("관측된 상태")
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if masked[i, j] != -1:
                axs[0].text(j, i, str(masked[i, j]), ha='center', va='center')

    # 예측 확률
    axs[1].imshow(pred, cmap='Reds')
    axs[1].set_title("예측된 지뢰 확률")
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            axs[1].text(j, i, f"{pred[i, j]:.2f}", ha='center', va='center', fontsize=8)

    # 실제 지뢰 위치
    axs[2].imshow((board == -1), cmap='binary')
    axs[2].set_title("실제 지뢰 위치")

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()

#실행 개선
if __name__ == "__main__":
    # 데이터 생성
    print("데이터 생성 중...")
    X, Y = create_dataset(5000)  # 더 큰 데이터셋
    
    # 데이터 분할
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # 모델 생성 및 학습
    print("\n모델 학습 중...")
    model = ImprovedMineDetector()
    train(model, X_train, Y_train, epochs=30, lr=1e-4)
    
    # 평가
    print("\n모델 평가:")
    evaluate(model, X_val, Y_val)
    
    # 시각화
    print("\n예측 결과 시각화:")
    visualize(model)
