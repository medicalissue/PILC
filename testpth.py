# test_eval.py
import os, argparse, random, math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from tokenizer import SoftVQModel, ModelArgs
from dataloader import create_dataloader


# ----------------------------
# utils
# ----------------------------
def to01(x):  # [-1,1] -> [0,1]
    return x.add(1).div(2).clamp(0, 1)

@torch.no_grad()
def pool_tokens(tokens, mode="mean"):
    if tokens.dim() == 3:
        return tokens.mean(dim=1)
    return tokens

@torch.no_grad()
def compute_retrieval(sim_matrix, topk=(1,5,10)):
    """sim_matrix: [N_text, N_img]"""
    N = sim_matrix.size(0)
    labels = torch.arange(N, device=sim_matrix.device)

    out = {}
    for k in topk:
        # T2I
        _, idx = sim_matrix.topk(k, dim=1)  # [N, k]
        match = (idx == labels[:, None]).any(dim=1).float().mean().item()
        out[f"t2i@{k}"] = match

        # I2T
        _, idx = sim_matrix.t().topk(k, dim=1)
        match = (idx == labels[:, None]).any(dim=1).float().mean().item()
        out[f"i2t@{k}"] = match
    return out


# ----------------------------
# main eval
# ----------------------------
@torch.no_grad()
def evaluate_ckpt(ckpt, data_root="/data/coco", split="val",
                  batch_size=64, out_dir="./eval_out", save_samples=10, pool="mean"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    state = torch.load(ckpt, map_location="cpu")
    model_args = state.get("args", None)
    if isinstance(model_args, dict):
        model_args = ModelArgs(**model_args)
    model = SoftVQModel(model_args)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval().to(device)

    # 데이터 로드
    loader = create_dataloader(
        data_root=data_root, dataset_type="coco", split=split,
        batch_size=batch_size, shuffle=False
    )

    all_text, all_vis = [], []
    all_images, all_recon = [], []

    # 1) 전체 val 임베딩 뽑기
    for images, texts in tqdm(loader, desc="Extract features"):
        images = images.to(device)
        out = model(images, texts)
        tf = F.normalize(pool_tokens(out["text_tokens"], pool), dim=-1)
        vf = F.normalize(pool_tokens(out["visual_tokens"], pool), dim=-1)
        all_text.append(tf.cpu())
        all_vis.append(vf.cpu())

        if len(all_images) < save_samples:
            all_images.append(images.cpu())
            all_recon.append(out["reconstructed"].cpu())

    all_text = torch.cat(all_text, dim=0)
    all_vis  = torch.cat(all_vis, dim=0)

    # 2) retrieval metric
    sim_matrix = all_text @ all_vis.t()
    metrics = compute_retrieval(sim_matrix, topk=(1,5,10))

    print("=== Retrieval (global val set) ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 3) reconstruction 샘플 저장 (10개만)
    ims = torch.cat(all_images, dim=0)[:save_samples]
    rec = torch.cat(all_recon,  dim=0)[:save_samples]
    grid = make_grid(torch.cat([to01(ims), to01(rec)], dim=0), nrow=save_samples)
    save_image(grid, os.path.join(out_dir, "recon_samples.png"))
    print(f"Saved recon samples to {out_dir}/recon_samples.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./checkpoints/checkpoint_epoch_7.pth")
    ap.add_argument("--data_root", type=str, default="/data/coco")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="./eval_out")
    ap.add_argument("--save_samples", type=int, default=10)
    ap.add_argument("--pool", type=str, default="mean")
    args = ap.parse_args()

    evaluate_ckpt(
        args.ckpt, args.data_root, args.split,
        args.batch_size, args.out_dir, args.save_samples, args.pool
    )
