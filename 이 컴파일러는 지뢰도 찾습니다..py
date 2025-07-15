import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, precision_recall_curve, auc, 
                            confusion_matrix)
import seaborn as sns
import os

# 한글 폰트 설정 (맑은 고딕)
plt.rc('font', family='Malgun Gothic')
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 게임 설정
BOARD_SIZE = 8  # 게임 보드 크기 (8x8)
NUM_MINES = 10   # 지뢰 개수

# 보드 생성 함수: 지뢰 위치를 무작위로 배치하고 주변 지뢰 개수 계산
def generate_board(size=BOARD_SIZE, num_mines=NUM_MINES):
    board = np.zeros((size, size), dtype=int)
    # 지뢰 위치 무작위 선택 (중복 없음)
    mines = np.random.choice(size * size, num_mines, replace=False)
    for idx in mines:
        x, y = divmod(idx, size)  # 1차원 인덱스를 2D 좌표로 변환
        board[x, y] = -1  # 지뢰는 -1로 표시
    
    # 각 셀의 주변 지뢰 개수 계산
    for x in range(size):
        for y in range(size):
            if board[x, y] == -1:  # 지뢰 셀은 건너뜀
                continue
            count = 0
            # 주변 8방향 검사 (경계 처리 포함)
            for i in range(max(0, x-1), min(size, x+2)):
                for j in range(max(0, y-1), min(size, y+2)):
                    if board[i, j] == -1:
                        count += 1
            board[x, y] = count  # 주변 지뢰 개수 저장
    return board

# 부분 가리기 함수: 일부 셀만 공개 (확률 p로 공개)
def mask_board(board, p=0.2):
    masked = np.full_like(board, -1)  # 모든 셀을 -1(숨김)으로 초기화
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # 지뢰가 아니고 확률 p에 의해 공개되는 셀
            if board[i, j] != -1 and np.random.rand() < p:
                masked[i, j] = board[i, j]  # 숫자 공개
    return masked

# Focal Loss: 클래스 불균형(지뢰가 적음)을 해결하기 위한 손실 함수
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 모델의 예측 확률
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss  # Focal Loss 계산
        
        # 손실 축소 방식 선택
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# U-Net 기반 지뢰 탐지 모델
class UNetMineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 인코더 (다운샘플링): 3단계
        self.enc1 = self._block(1, 64)  # 입력 채널 1, 출력 채널 64
        self.pool1 = nn.MaxPool2d(2)    # 1/2 다운샘플링
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        
        # 병목 계층: 가장 낮은 해상도에서 특징 추출
        self.bottleneck = self._block(256, 512)
        
        # 디코더 (업샘플링): 3단계 (skip connection 사용)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 업샘플링
        self.dec3 = self._block(512, 256)  # 인코더의 특징맵과 채널 결합(concat)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # 출력 계층: 1채널 출력 (지뢰 확률)
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # 확률값으로 변환
        
        # 가중치 초기화
        self._initialize_weights()
    
    # 기본 블록: 두 개의 합성곱 레이어, 배치 정규화, ReLU
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    # 가중치 초기화: Xavier 초기화
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 인코더 경로
        enc1 = self.enc1(x)            # [B, 64, H, W]
        enc2 = self.enc2(self.pool1(enc1)) # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2)) # [B, 256, H/4, W/4]
        
        # 병목 계층
        bottleneck = self.bottleneck(self.pool2(enc3))  # [B, 512, H/8, W/8]
        
        # 디코더 경로 (skip connection)
        dec3 = self.up3(bottleneck)    # [B, 256, H/4, W/4]
        dec3 = torch.cat((dec3, enc3), dim=1)  # [B, 512, H/4, W/4]
        dec3 = self.dec3(dec3)         # [B, 256, H/4, W/4]
        
        dec2 = self.up2(dec3)          # [B, 128, H/2, W/2]
        dec2 = torch.cat((dec2, enc2), dim=1)  # [B, 256, H/2, W/2]
        dec2 = self.dec2(dec2)         # [B, 128, H/2, W/2]
        
        dec1 = self.up1(dec2)          # [B, 64, H, W]
        dec1 = torch.cat((dec1, enc1), dim=1)  # [B, 128, H, W]
        dec1 = self.dec1(dec1)         # [B, 64, H, W]
        
        return self.sigmoid(self.out(dec1))  # [B, 1, H, W]

# 사용자 정의 데이터셋 클래스
class MineDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, transform=True):
        self.boards = []    # 부분 가려진 보드 저장
        self.targets = []   # 지뢰 위치(이진 맵) 저장
        self.transform = transform  # 데이터 증강 적용 여부
        
        # n_samples 개수만큼 데이터 생성
        for _ in range(n_samples):
            full = generate_board()
            p = np.random.uniform(0.1, 0.5)  # 공개 확률 무작위 설정
            obs = mask_board(full, p)
            self.boards.append(obs)
            # 지뢰 위치 이진 맵 생성 (지뢰: 1, 안전: 0)
            self.targets.append((full == -1).astype(np.float32))
    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        board = self.boards[idx]
        target = self.targets[idx]
        
        # 온라인 데이터 증강: 훈련 시에만 적용
        if self.transform:
            # 랜덤 회전 (0, 90, 180, 270도)
            k = np.random.choice(4)
            board = np.rot90(board, k)
            target = np.rot90(target, k)
            
            # 50% 확률로 수평 반전
            if np.random.rand() > 0.5:
                board = np.fliplr(board)
                target = np.fliplr(target)
        
        # 텐서로 변환 (채널 차원 추가)
        return (torch.tensor(board).unsqueeze(0).float(), 
                torch.tensor(target).unsqueeze(0).float())

# 모델 학습 함수
def train(model, train_loader, val_loader, epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Focal Loss (클래스 불균형 해결)
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    
    # AdamW 옵티마이저 (가중치 감쇠 포함)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine Annealing 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5  # 조기 종료를 위한 인내심 설정
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # 학습 루프 (tqdm으로 진행률 표시)
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # 순전파 -> 역전파 -> 가중치 업데이트
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # 배치 손실 누적
        
        # 검증 루프
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred_val = model(x_val)
                val_loss += criterion(pred_val, y_val).item()
        
        # 평균 손실 계산
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 학습률 조정
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # 조기 종료 및 모델 저장 로직
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')  # 최적 모델 저장
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # 최고 성능 모델 로드
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 모델 평가 함수
def evaluate(model, loader, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []   # 모든 예측 확률 저장
    all_targets = [] # 모든 정답 라벨 저장
    
    # 예측 및 라벨 수집
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    
    # 리스트를 텐서로 변환
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # 이진 분류: 확률을 threshold 기준으로 변환
    preds_binary = (preds > threshold).float()
    
    # 성능 지표 계산
    accuracy = (preds_binary == targets).float().mean()  # 정확도
    precision = precision_score(targets.view(-1), preds_binary.view(-1))  # 정밀도
    recall = recall_score(targets.view(-1), preds_binary.view(-1))        # 재현율
    f1 = f1_score(targets.view(-1), preds_binary.view(-1))                # F1 점수
    
    # ROC-AUC 및 PR-AUC
    roc_auc = roc_auc_score(targets.view(-1), preds.view(-1))
    precision_curve, recall_curve, _ = precision_recall_curve(targets.view(-1), preds.view(-1))
    pr_auc = auc(recall_curve, precision_curve)  # Precision-Recall 곡선 아래 면적
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(targets.view(-1), preds_binary.view(-1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Mine'], 
                yticklabels=['Safe', 'Mine'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 결과 출력
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

# 예측 결과 시각화 함수
def visualize(model, num_samples=9, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()
    
    # 3행 x num_samples 열의 서브플롯 생성
    fig, axs = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    for i in range(num_samples):
        # 새 게임 보드 생성
        board = generate_board()
        masked = mask_board(board)  # 부분 가려진 보드
        input_tensor = torch.tensor(masked).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # 모델 예측
        with torch.no_grad():
            pred = model(input_tensor).squeeze().cpu().numpy()  # 지뢰 확률 맵
        
        pred_binary = (pred > threshold).astype(float)  # 이진 분류 결과
        actual_mines = (board == -1).astype(float)      # 실제 지뢰 위치
        
        # [행 0] 관측된 보드 시각화
        axs[0, i].imshow(masked, cmap='gray', vmin=-1, vmax=8)
        axs[0, i].set_title(f"Observed #{i+1}")
        # 공개된 셀에 숫자 표시
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if masked[x, y] != -1:
                    axs[0, i].text(y, x, str(masked[x, y]), 
                                  ha='center', va='center', 
                                  color='black' if masked[x, y] > 0 else 'white')
        
        # [행 1] 예측 확률 맵 시각화
        axs[1, i].imshow(pred, cmap='Reds', vmin=0, vmax=1)
        axs[1, i].set_title(f"Prediction #{i+1}")
        # 각 셀의 확률값 표시
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                axs[1, i].text(y, x, f"{pred[x, y]:.2f}", 
                              ha='center', va='center', 
                              fontsize=8, 
                              color='black' if pred[x, y] < 0.5 else 'white')
        
        # [행 2] 오분류 시각화 (RGB 혼합)
        combined = np.zeros((BOARD_SIZE, BOARD_SIZE, 3))
        combined[..., 0] = actual_mines  # 실제 지뢰: 빨간색
        combined[..., 1] = pred_binary   # 예측 지뢰: 녹색
        
        # 오분류 계산
        fp = (pred_binary == 1) & (actual_mines == 0)  # False Positive
        fn = (pred_binary == 0) & (actual_mines == 1)  # False Negative
        
        # FP(거짓 양성): 파란색으로 표시
        combined[fp, 0] = 0
        combined[fp, 1] = 0
        combined[fp, 2] = 1
        
        # FN(거짓 음성): 노란색으로 표시
        combined[fn, 0] = 1
        combined[fn, 1] = 1
        combined[fn, 2] = 0
        
        axs[2, i].imshow(combined)
        axs[2, i].set_title(f"Comparison #{i+1}")
        
        # 오분류 수 표시
        fp_count = fp.sum()
        fn_count = fn.sum()
        axs[2, i].text(0.5, -0.15, f"FP: {fp_count}, FN: {fn_count}", 
                      ha='center', va='top', transform=axs[2, i].transAxes,
                      bbox=dict(facecolor='white', alpha=0.7))
    
    # 축 라벨 제거
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')  # 이미지 저장
    plt.show()

# 메인 실행 블록
if __name__ == "__main__":
    # 1. 데이터셋 생성
    print("데이터셋 생성 중...")
    full_dataset = MineDataset(5000, transform=True)  # 5000개 샘플, 증강 적용
    
    # 2. 데이터 분할 (80% 훈련, 20% 검증)
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # 3. 데이터 로더 생성 (배치 크기 32, 4개 워커)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # 4. 모델 생성
    print("\n모델 생성 중...")
    model = UNetMineDetector()
    
    # 5. 모델 학습
    print("\n모델 학습 시작...")
    trained_model = train(model, train_loader, val_loader, epochs=50, lr=1e-4)
    
    # 6. 모델 평가
    print("\n모델 평가:")
    metrics = evaluate(trained_model, val_loader)
    
    # 7. 예측 결과 시각화 (3x3 그리드)
    print("\n예측 결과 시각화:")
    visualize(trained_model, num_samples=9)