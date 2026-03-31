import torch
import torch.nn as nn


# --------- Activations ----------
class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "waveact":
        return WaveAct()
    raise ValueError("activation must be one of: relu | tanh | silu | waveact")


# --------- Attention ----------
class MultiHeadSelfAttention(nn.Module):
    """
    Custom attention implementation that stays on the maths kernel so that
    higher-order autograd required by PDE losses remains available.
    """

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by the number of heads")
        self.heads = heads
        self.d_head = d_model // heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, S, 3, self.heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        ctx = torch.matmul(attn, v)  # (B, H, S, d_head)

        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.proj(ctx)


# --------- Feed Forward ----------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, activation: str):
        super().__init__()
        d_ff = 2 * d_model
        act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Linear(d_ff, d_model),
        )
        # Zero init last layer for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# --------- Encoder Layer (pre-norm) ----------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, activation: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.ln1(x))
        x = x + y
        y = self.ffn(self.ln2(x))
        x = x + y
        return x


# --------- Encoder Stack ----------
class Encoder(nn.Module):
    def __init__(self, d_model: int, N: int, heads: int, activation: str):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, activation) for _ in range(N)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# --------- Model Wrapper ----------
class SetPinns(nn.Module):
    def __init__(
        self,
        d_out: int,
        d_model: int,
        d_hidden: int,
        N: int,
        heads: int,
        in_dim: int,
        activation: str,
    ):
        super().__init__()
        self.linear_emb = nn.Linear(in_dim, d_model)
        self.encoder = Encoder(d_model, N, heads, activation)
        act = get_activation(activation)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            act,
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, *v: torch.Tensor) -> torch.Tensor:
        src = torch.cat(v, dim=-1) if len(v) > 1 else v[0]  # (B,S,in_dim)
        x = self.linear_emb(src)
        x = self.encoder(x)
        return self.head(x)


# --------- Quick check ----------
if __name__ == "__main__":
    model = SetPinns(d_out=1, d_model=128, d_hidden=64, N=2, heads=4, activation="silu")
    B, S = 16, 32
    x = torch.randn(B, S, 1)
    t = torch.randn(B, S, 1)
    y = model(x, t)
    print(y.shape)  # (B,S,1)
