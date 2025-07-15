import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ImprovedMathFormulaModel:
    def __init__(self, formula_path, image_root):
        self.formula_path = formula_path
        self.image_root = image_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._build_vocab()
        self._prepare_data()
        self._setup_transforms()
        self._initialize_models()
        self._setup_optimizer()
    
    def _build_vocab(self):
        """빌드 어휘 및 토큰 매핑"""
        tokens = set()
        with open(self.formula_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.update(line.split())
        
        # 특수 토큰 추가
        extra_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        tokens.update(extra_tokens)
        
        # 정렬 및 매핑 생성
        self.vocab = sorted(tokens)
        self.token_to_idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        self.pad_idx = self.token_to_idx["<PAD>"]
    
    def _prepare_data(self):
        """데이터 준비"""
        self.data = []
        with open(self.formula_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                img_name = f"00001 ({i+1}).png"
                tokens = ["<SOS>"] + line.split() + ["<EOS>"]
                self.data.append((img_name, tokens))
    
    def _setup_transforms(self):
        """이미지 전처리 설정"""
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 512)),  # 더 높은 해상도
            transforms.ColorJitter(contrast=0.2),  # 콘트라스트 조정으로 일반화 향상
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화
        ])
    
    def _initialize_models(self):
        """모델 초기화"""
        # 향상된 CNN 인코더
        self.encoder = EnhancedCNN().to(self.device)
        
        # 인코더 출력 차원 계산
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 512).to(self.device)
            dummy_output = self.encoder(dummy_input)
            encoded_dim = dummy_output.shape[1]
        
        # 향상된 RNN 디코더
        self.decoder = EnhancedRNN(
            input_dim=encoded_dim,
            hidden_dim=512,
            vocab_size=len(self.vocab),
            embedding_dim=256,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
    
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 설정"""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
    
    def train(self, num_epochs=10, batch_size=8):
        """모델 학습"""
        dataset = MathFormulaDataset(
            self.data, 
            self.image_root, 
            self.token_to_idx, 
            transform=self.transform
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self._collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Teacher forcing 적용 (75% 확률)
                use_teacher_forcing = torch.rand(1).item() < 0.75
                
                # 인코더 특징 추출
                features = self.encoder(images)
                
                if use_teacher_forcing:
                    # Teacher forcing 사용 시
                    decoder_input = labels[:, :-1]
                    decoder_target = labels[:, 1:]
                    
                    outputs = self.decoder(features, decoder_input)
                    loss = self.criterion(
                        outputs.reshape(-1, len(self.vocab)), 
                        decoder_target.reshape(-1)
                    )
                else:
                    # Teacher forcing 미사용 시 (자기 회귀 생성)
                    loss = 0
                    batch_size = images.size(0)
                    input_seq = torch.tensor(
                        [self.token_to_idx["<SOS>"]] * batch_size,
                        dtype=torch.long,
                        device=self.device
                    ).unsqueeze(1)
                    
                    for i in range(labels.size(1) - 1):
                        outputs = self.decoder(features, input_seq)
                        _, topi = outputs[:, -1, :].topk(1)
                        input_seq = torch.cat([input_seq, topi], dim=1)
                        
                        # 손실 계산 (현재 타임스텝만)
                        current_loss = self.criterion(
                            outputs[:, -1, :], 
                            labels[:, i+1]
                        )
                        loss += current_loss
                    
                    loss /= (labels.size(1) - 1)
                
                # 역전파 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            
            # 모델 저장 (성능 향상 시)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'vocab': self.vocab,
                    'token_to_idx': self.token_to_idx
                }, 'best_model.pth')
            
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def predict(self, image_path, max_len=50, beam_size=3):
        """이미지에서 수식 예측 (빔 서치 적용)"""
        self.encoder.eval()
        self.decoder.eval()
        
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("L")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # 특징 추출
        with torch.no_grad():
            features = self.encoder(image)
        
        # 빔 서치 초기화
        start_token = self.token_to_idx["<SOS>"]
        beams = [([start_token], 0)]  # (토큰 시퀀스, 로그 확률)
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score in beams:
                # 시�언스가 이미 EOS로 끝나면 후보에 그대로 추가
                if seq[-1] == self.token_to_idx["<EOS>"]:
                    candidates.append((seq, score))
                    continue
                
                # 디코더 입력 준비
                input_seq = torch.tensor(seq, device=self.device).unsqueeze(0)
                
                # 디코더 실행
                with torch.no_grad():
                    output = self.decoder(features, input_seq)
                    log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                    topk_probs, topk_tokens = log_probs.topk(beam_size)
                
                # 새로운 후보 생성
                for i in range(beam_size):
                    new_seq = seq + [topk_tokens[0, i].item()]
                    new_score = score + topk_probs[0, i].item()
                    candidates.append((new_seq, new_score))
            
            # 상위 beam_size개 후보 선택
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # 모든 시퀀스가 EOS로 끝나면 종료
            if all(seq[-1] == self.token_to_idx["<EOS>"] for seq, _ in beams):
                break
        
        # 가장 확률이 높은 시퀀스 선택
        best_seq = beams[0][0]
        tokens = [self.idx_to_token[idx] for idx in best_seq]
        
        # 특수 토큰 제거
        tokens = [t for t in tokens if t not in ["<SOS>", "<EOS>", "<PAD>"]]
        return " ".join(tokens)
    
    def _collate_fn(self, batch):
        """데이터 배치 처리"""
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_idx)
        return images, labels

class EnhancedCNN(nn.Module):
    """향상된 CNN 인코더"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x512 -> 64x256
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x256 -> 32x128
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 32x128 -> 16x128
            
            # Attention 준비를 위한 추가 컨볼루션
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # 특징 맵 평탄화
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # 높이를 1로 축소
        
    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)  # [batch, 512, 1, W]
        x = x.squeeze(2)  # [batch, 512, W]
        x = x.permute(0, 2, 1)  # [batch, W, 512] (시퀀스, 특징)
        return x

class EnhancedRNN(nn.Module):
    """향상된 RNN 디코더 (어텐션 메커니즘 포함)"""
    def __init__(self, input_dim, hidden_dim, vocab_size, embedding_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.LSTM(
            embedding_dim + input_dim,  # 어텐션 적용 시 컨텍스트 벡터 포함
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_out, captions):
        # 임베딩
        embedded = self.dropout(self.embedding(captions))  # [batch, seq_len, emb_dim]
        
        # 어텐션 적용
        outputs = []
        hidden = None
        for i in range(embedded.size(1)):
            # 현재 단계의 임베딩
            current_embed = embedded[:, i:i+1, :]
            
            # 어텐션 계산
            context, _ = self.attention(hidden, encoder_out) if hidden is not None else (encoder_out.mean(1, keepdim=True), None)
            
            # RNN 입력 준비
            rnn_input = torch.cat([current_embed, context], dim=-1)
            
            # RNN 실행
            out, hidden = self.rnn(rnn_input, hidden)
            
            # 출력 생성
            out = self.fc(self.dropout(out.squeeze(1)))
            outputs.append(out)
        
        # 모든 출력을 하나의 텐서로 결합
        outputs = torch.stack(outputs, dim=1)
        return outputs

class Attention(nn.Module):
    """어텐션 메커니즘"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(512, hidden_dim)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, hidden, encoder_out):
        if hidden is None:
            # 초기 단계에서는 균등한 어텐션 가정
            batch_size = encoder_out.size(0)
            seq_len = encoder_out.size(1)
            attention = torch.ones(batch_size, seq_len, device=encoder_out.device) / seq_len
            context = torch.bmm(attention.unsqueeze(1), encoder_out)
            return context, attention
        
        # hidden: [num_layers, batch, hidden_dim]
        # 마지막 레이어의 hidden state만 사용
        hidden = hidden[0][-1].unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 어텐션 스코어 계산
        encoder_proj = self.encoder_proj(encoder_out)  # [batch, seq_len, hidden_dim]
        decoder_proj = self.decoder_proj(hidden)  # [batch, 1, hidden_dim]
        
        scores = self.v(self.tanh(encoder_proj + decoder_proj)).squeeze(2)  # [batch, seq_len]
        attention = F.softmax(scores, dim=1)
        
        # 컨텍스트 벡터 계산
        context = torch.bmm(attention.unsqueeze(1), encoder_out)  # [batch, 1, 512]
        return context, attention

# 사용 예시
if __name__ == "__main__":
    formula_path = "C:/Users/user/Downloads/im2markup-master/im2markup-master/data/sample/formulas.lst"
    image_root = "C:/Users/user/Downloads/im2markup-master/im2markup-master/data/sample/images"
    
    # 모델 생성 및 학습
    model = ImprovedMathFormulaModel(formula_path, image_root)
    model.train(num_epochs=10, batch_size=16)
    
    # 예측 테스트
    test_img_path = os.path.join(image_root, "bc13232098.png")
    predicted_formula = model.predict(test_img_path)
    print(f"Predicted Formula: {predicted_formula}")
