import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

BOARD_SIZE = 8
NUM_MINES = 10

# 전체 보드 생성 (숫자/지뢰 포함)
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

# 일부 셀만 공개된 입력 상태 생성
def mask_board(board, p=0.2):
    masked = np.full_like(board, -1)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] != -1 and np.random.rand() < p:
                masked[i, j] = board[i, j]
    return masked

# 학습 데이터 생성
def create_dataset(n=1000):
    X, Y = [], []
    for _ in tqdm(range(n)):
        full = generate_board()
        obs = mask_board(full)
        X.append(obs)
        Y.append((full == -1).astype(np.float32))
    X = torch.tensor(X).unsqueeze(1).float()  # (B, 1, H, W)
    Y = torch.tensor(Y).unsqueeze(1).float()
    return X, Y

# CNN 모델 정의
class MineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 학습 루프
def train(model, X, Y, epochs=10, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 예측 결과 시각화
def visualize(model):
    model.eval()
    board = generate_board()
    masked = mask_board(board)
    input_tensor = torch.tensor(masked).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        pred = model(input_tensor).squeeze().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 관측된 보드
    axs[0].imshow(masked, cmap='gray', vmin=-1, vmax=8)
    axs[0].set_title("📘 관측된 상태")
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if masked[i, j] != -1:
                axs[0].text(j, i, str(masked[i, j]), ha='center', va='center')

    # 예측 확률
    axs[1].imshow(pred, cmap='Reds')
    axs[1].set_title("🤖 예측된 지뢰 확률")
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            axs[1].text(j, i, f"{pred[i, j]:.2f}", ha='center', va='center', fontsize=8)

    # 실제 지뢰 위치
    axs[2].imshow((board == -1), cmap='binary')
    axs[2].set_title("🎯 실제 지뢰 위치")

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()

# 실행
if __name__ == "__main__":
    X, Y = create_dataset(1000)
    model = MineDetector()
    train(model, X, Y, epochs=10)
    visualize(model)
