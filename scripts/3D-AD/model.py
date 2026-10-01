"""
3D Student-Teacher (3D-ST) 모델 정의.

논문 수식 대응
- 식 (1)  G(p, p_j) = (p - p_j) ⊕ ||p - p_j||_2        -> geometric_features()
- LFA 블록 (Fig. 4b)                                   -> LFA
- 잔차 블록 (Fig. 4a)                                   -> ResidualBlock
- 교사/학생 특징 추출기 (동일 아키텍처)                  -> FeatureExtractor
- 식 (2)  R(p) = ∪_{l=0..L} knn^l(p)                   -> receptive_field_masks()
- 디코더 D : R^d -> R^{3×m}                             -> Decoder
- 식 (3)  L_C = 1/|Q| Σ Chamfer(D(f_p), R̄(p))           -> chamfer_distance(), pretrain loss
- 식 (5)  L_ST = 1/|P| Σ ||f_S - (f_T - μ) diag(σ)^-1||² -> student_loss()
"""
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 식 (1): 국소 기하 특징. 좌표 차이만 사용하므로 평행이동 불변.
# --------------------------------------------------------------------------- #
def geometric_features(points: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """
    points : (n, 3)   정규화된 좌표 (×1/s 적용 후)
    knn_idx: (n, k)   각 점의 k-최근접 이웃 인덱스
    return : (n, k, 4) = [p - p_j, ||p - p_j||]
    """
    diff = points.unsqueeze(1) - points[knn_idx]          # (n, k, 3)
    dist = diff.norm(dim=-1, keepdim=True)                # (n, k, 1)
    return torch.cat([diff, dist], dim=-1)


class SharedMLP(nn.Module):
    """단일 dense layer + LeakyReLU(0.2) (논문 4.1절)."""

    def __init__(self, c_in, c_out, slope=0.2):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)
        self.act = nn.LeakyReLU(slope)

    def forward(self, x):
        return self.act(self.fc(x))


# --------------------------------------------------------------------------- #
# LFA 블록 (Fig. 4b)
#   G(p,p_j) --sharedMLP--> (n,k,d_LFA)
#   ⊕ 이웃 입력 특징 {f_pj} (n,k,d_LFA)
#   --평균 풀링(이웃 축)--> (n, 2·d_LFA)
# --------------------------------------------------------------------------- #
class LFA(nn.Module):
    def __init__(self, d_lfa):
        super().__init__()
        self.geo_mlp = SharedMLP(4, d_lfa)

    def forward(self, feats, geo, knn_idx):
        """
        feats  : (n, d_lfa)  입력 특징
        geo    : (n, k, 4)   식 (1)
        knn_idx: (n, k)
        return : (n, 2·d_lfa)
        """
        g = self.geo_mlp(geo)                    # (n, k, d_lfa)
        f_nb = feats[knn_idx]                    # (n, k, d_lfa)
        return torch.cat([g, f_nb], dim=-1).mean(dim=1)


# --------------------------------------------------------------------------- #
# 잔차 블록 (Fig. 4a): sharedMLP -> LFA -> LFA -> sharedMLP, 입력에 더함.
# 차원(가정; 본문에 Fig.4 수치가 없어 RandLA-Net 관례를 따름):
#   d -> d/4 -> LFA(d_LFA=d/4) -> d/2 -> LFA(d_LFA=d/2) -> d -> d
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        assert d % 4 == 0
        self.mlp_in = SharedMLP(d, d // 4)
        self.lfa1 = LFA(d // 4)          # out: d/2
        self.lfa2 = LFA(d // 2)          # out: d
        self.mlp_out = SharedMLP(d, d)

    def forward(self, f, geo, knn_idx):
        h = self.mlp_in(f)
        h = self.lfa1(h, geo, knn_idx)
        h = self.lfa2(h, geo, knn_idx)
        h = self.mlp_out(h)
        return f + h


class FeatureExtractor(nn.Module):
    """
    교사 T와 학생 S가 공유하는 아키텍처.
    f_p = 0 으로 초기화 -> 잔차 블록 × B -> 단일 은닉층 MLP (d -> d -> d).
    LFA 개수 L = 2·B  (식 2의 hop 수)
    """

    def __init__(self, d=64, num_blocks=4):
        super().__init__()
        self.d = d
        self.blocks = nn.ModuleList([ResidualBlock(d) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.LeakyReLU(0.2), nn.Linear(d, d)
        )

    @property
    def num_lfa(self):
        return 2 * len(self.blocks)

    def forward(self, points, knn_idx):
        """
        points : (n, 3) 정규화된 좌표, knn_idx: (n, k)
        return : (n, d) 점별 dense descriptor
        """
        geo = geometric_features(points, knn_idx)       # 모든 블록에서 재사용
        f = points.new_zeros(points.shape[0], self.d)   # f_p = 0
        for blk in self.blocks:
            f = blk(f, geo, knn_idx)
        return self.head(f)


# --------------------------------------------------------------------------- #
# 디코더 D : R^d -> R^{3×m}
# 입력 d, 은닉 128 ×2 (LeakyReLU 0.05), 출력 m=1024 점
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    def __init__(self, d=64, m=1024, hidden=128):
        super().__init__()
        self.m = m
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.05),
            nn.Linear(hidden, 3 * m),
        )

    def forward(self, f):                 # (B, d)
        return self.net(f).view(-1, self.m, 3)


# --------------------------------------------------------------------------- #
# 식 (2): 수용 영역. kNN 그래프를 L hop 확장.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def receptive_field_masks(knn_idx: torch.Tensor, query_idx: torch.Tensor, num_hops: int):
    """
    knn_idx  : (n, k)
    query_idx: (B,)  샘플링된 점 집합 Q
    return   : (B, n) bool,  mask[b, i] = 점 i ∈ R(q_b)
    """
    n = knn_idx.shape[0]
    B = query_idx.shape[0]
    mask = torch.zeros(B, n, dtype=torch.bool, device=knn_idx.device)
    mask[torch.arange(B, device=knn_idx.device), query_idx] = True   # knn^0 = {p}
    frontier = mask.clone()
    for _ in range(num_hops):
        new = torch.zeros_like(mask)
        for b in range(B):
            nb = knn_idx[frontier[b]].reshape(-1)          # knn^l 의 이웃들
            new[b, nb] = True
        frontier = new & ~mask                               # 새로 추가된 점만 다음 hop
        mask |= new
        if not frontier.any():
            break
    return mask


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    양방향 Chamfer 거리 (각 방향: 최근접 제곱거리의 평균, 두 방향 합).
    a: (m, 3), b: (r, 3)  크기가 달라도 됨. 쌍별 거리 행렬 (m × r) 사용.
    """
    d = torch.cdist(a, b).pow(2)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def pretrain_loss(teacher, decoder, points, knn_idx, num_queries=16, max_rf_points=None):
    """
    식 (3): L_C(D) = 1/|Q| Σ_{p∈Q} Chamfer(D(f_p), R̄(p)),  R̄(p) = R(p) - mean(R(p))
    교사 T와 디코더 D를 함께 학습.
    """
    feats = teacher(points, knn_idx)                                  # (n, d)
    n = points.shape[0]
    q = torch.randperm(n, device=points.device)[:num_queries]         # Q
    masks = receptive_field_masks(knn_idx, q, teacher.num_lfa)        # (|Q|, n)
    recon = decoder(feats[q])                                         # (|Q|, m, 3)

    loss = 0.0
    rf_sizes = []
    for b in range(q.shape[0]):
        rf = points[masks[b]]                                         # R(p)
        if max_rf_points is not None and rf.shape[0] > max_rf_points:
            rf = rf[torch.randperm(rf.shape[0], device=rf.device)[:max_rf_points]]
        rf = rf - rf.mean(dim=0, keepdim=True)                        # R̄(p)
        loss = loss + chamfer_distance(recon[b], rf)
        rf_sizes.append(int(masks[b].sum()))
    return loss / q.shape[0], rf_sizes


# --------------------------------------------------------------------------- #
# 식 (5) 및 추론 시 이상 점수
# --------------------------------------------------------------------------- #
def normalize_teacher(f_t, mu, sigma):
    """(f_T - μ) diag(σ)^-1  ==  (f_T - μ) / σ (원소별)"""
    return (f_t - mu) / sigma


def student_loss(f_s, f_t, mu, sigma):
    """L_ST(S) = 1/|P| Σ_p ||f_S - (f_T - μ)/σ||²"""
    return (f_s - normalize_teacher(f_t, mu, sigma)).pow(2).sum(dim=1).mean()


def anomaly_scores(f_s, f_t, mu, sigma):
    """점별 이상 점수 = ||f_S - (f_T - μ)/σ||_2"""
    return (f_s - normalize_teacher(f_t, mu, sigma)).norm(dim=1)
