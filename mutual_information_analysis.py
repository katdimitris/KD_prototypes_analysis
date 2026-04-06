import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import timm
import torch
from networkx.algorithms.clique import enumerate_all_cliques
from timm import create_model
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from timm.models import register_model


@register_model
def vit_tiny_patch16_dinov3(pretrained: bool = False, **kwargs):
    return timm.create_model(
        "vit_small_patch16_dinov3",
        pretrained=pretrained,
        embed_dim=192,
        num_heads=3,
        **kwargs,
    )


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def print_stats(x: torch.Tensor, name: str):
    print(
        f"{name}: "
        f"shape={tuple(x.shape)} "
        f"mean={x.mean().item():.12f} "
        f"std={x.std().item():.12f} "
        f"min={x.min().item():.12f} "
        f"max={x.max().item():.12f} "
        f"norm={x.norm().item():.12f}"
    )


@torch.no_grad()
def compute_bn_stats(model, dataloader, n_tokens, path="teacher_val_bn_stats.pt", device='cuda', use_projector=False):
    stats_path = Path(args.outdir) / Path(args.exp_name) / "cache" / path
    ensure_dir(stats_path.parent)

    if stats_path.exists():
        print(f"[cache] loading full-validation feature stats from {stats_path}")
        obj = torch.load(stats_path, map_location="cpu")
        mean_cpu = obj["mean"].float().cpu()
        std_cpu = obj["std"].float().cpu()
        return mean_cpu, std_cpu

    print("[cache miss] computing full-validation feature stats...")
    feat_sum = None
    feat_sq_sum = None
    total_count = 0
    for idx, (imgs, _) in enumerate(tqdm(dataloader, desc="compute stats")):
        imgs = imgs.to(device, non_blocking=True)
        tok = model.forward_features(imgs)
        if use_projector:
            tok = model.projector(tok)
        tok = tok.reshape(-1, tok.shape[-1]).float()

        batch_sum = tok.sum(dim=0)
        batch_sq_sum = (tok * tok).sum(dim=0)
        batch_count = tok.shape[0]

        if feat_sum is None:
            feat_sum = batch_sum.detach().cpu()
            feat_sq_sum = batch_sq_sum.detach().cpu()
        else:
            feat_sum += batch_sum.detach().cpu()
            feat_sq_sum += batch_sq_sum.detach().cpu()

        total_count += batch_count
        if total_count > n_tokens:
            break
    if total_count == 0:
        raise RuntimeError("No tokens found while computing stats.")

    mean_cpu = feat_sum / total_count
    var_cpu = feat_sq_sum / total_count - mean_cpu * mean_cpu
    var_cpu = torch.clamp(var_cpu, min=0.0)
    std_cpu = torch.sqrt(var_cpu + 1e-5)

    torch.save(
        {
            "mean": mean_cpu,
            "std": std_cpu,
        },
        stats_path,
    )
    return mean_cpu, std_cpu


@torch.no_grad()
def compute_tensor_bn_stats(x, path):
    stats_path = Path(args.outdir) / Path(args.exp_name) / "cache" / path
    ensure_dir(stats_path.parent)

    if stats_path.exists():
        print(f"[cache] loading tensor stats from {stats_path}")
        obj = torch.load(stats_path, map_location="cpu")
        mean_cpu = obj["mean"].float().cpu()
        std_cpu = obj["std"].float().cpu()
        return mean_cpu, std_cpu

    print(f"[cache miss] computing tensor stats for {path}...")
    x = x.float().cpu()
    mean_cpu = x.mean(dim=0)
    var_cpu = x.var(dim=0, unbiased=False)
    var_cpu = torch.clamp(var_cpu, min=0.0)
    std_cpu = torch.sqrt(var_cpu + 1e-5)

    torch.save(
        {
            "mean": mean_cpu,
            "std": std_cpu,
        },
        stats_path,
    )
    return mean_cpu, std_cpu


def get_dataloader():
    ds = datasets.ImageFolder(
        args.imagenet,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    )
    perm = torch.randperm(len(ds))
    ds = Subset(ds, perm)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return dl


@torch.no_grad()
def sample_tokens(model, dataloader, indices, path, device="cuda:0", use_projector=False):
    cache_path = Path(args.outdir) / Path(args.exp_name) / "cache" / path

    if cache_path.exists():
        print(f"[cache] loading tokens from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    model.eval()
    indices = indices.cpu()
    out = []
    ptr = 0
    token_offset = 0

    for imgs, _ in tqdm(dataloader):
        if ptr == len(indices):
            break

        imgs = imgs.to(device, non_blocking=True)
        tok = model.forward_features(imgs)
        if use_projector:
            tok = model.projector(tok)
        B, T, D = tok.shape
        batch_tokens = B * T
        upper = token_offset + batch_tokens

        end = ptr
        while end < len(indices) and indices[end] < upper:
            end += 1

        if end > ptr:
            local_idx = (indices[ptr:end] - token_offset).to(tok.device)
            flat_tok = tok.reshape(batch_tokens, D)
            out.append(flat_tok[local_idx].cpu())

        ptr = end
        token_offset = upper

    if ptr != len(indices):
        raise RuntimeError("Not all requested token indices were found.")

    z = torch.cat(out, dim=0).contiguous()
    torch.save(z, cache_path)

    return z


def get_checkpoint_paths(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    paths = []
    for epoch in range(10, 301, 10):
        p = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        paths.append((epoch, p))
    return paths


def load_student_and_proto(student_model_name: str, ckpt_path, device: str):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    clean = {}
    for k, v in state.items():
        while k.startswith(("module.", "student.", "teacher.")):
            for p in ("module.", "student.", "teacher."):
                if k.startswith(p):
                    k = k[len(p):]
        clean[k] = v

    student = timm.create_model(student_model_name, pretrained=False).to(device)
    student.load_state_dict(clean, strict=False)

    W = clean["proto_proj_module.projectors.2.projs.2.weight"].float()
    student.projector = nn.Linear(W.shape[1], W.shape[0], bias=False).to(device)
    student.projector.weight.data.copy_(W.to(device))

    prototypes = clean["proto_proj_module.prototypes.2.protos.2"].float().to(device)

    return student.eval(), prototypes.cpu()


def kernel(x, y, eps=1e-12):

    if args.kernel == 'cosine':
        x_n = x / x.norm(dim=1, keepdim=True).clamp_min(eps)
        y_n = y / y.norm(dim=1, keepdim=True).clamp_min(eps)

        return x_n @ y_n.T
    else:
        # x2 = (x * x).sum(dim=1, keepdim=True)
        # y2 = (y * y).sum(dim=1, keepdim=True).T
        # d2 = x2 + y2 - 2.0 * (x @ y.T)
        # d2 = torch.clamp(d2, min=0.0)
        #
        # return -d2
        dist = torch.cdist(x, y, p=2)
        dist_sq = dist.pow(2) / x.shape[1]
        return -dist_sq

def assignment_probs(Z, P):
    C = kernel(Z, P)
    probs = torch.softmax(C / args.sigma, dim=1)
    return probs, C


def entropy_metrics(probs, eps=1e-12):
    H_A_given_Z = -(probs * probs.clamp_min(eps).log()).sum(dim=1).mean()
    marginal = probs.mean(dim=0)
    H_A = -(marginal * marginal.clamp_min(eps).log()).sum()
    I_Z_A = H_A - H_A_given_Z

    return {
        "H_A": H_A.item(),
        "H_A_given_Z": H_A_given_Z.item(),
        "I_Z_A": I_Z_A.item(),
    }


def teacher_student_assignment_metrics(teacher_probs, student_probs, eps=1e-12):
    joint = (teacher_probs.T @ student_probs) / teacher_probs.shape[0]
    p_t = joint.sum(dim=1, keepdim=True)
    p_s = joint.sum(dim=0, keepdim=True)

    I_A_T_A_S = (
        joint * (
            joint.clamp_min(eps).log()
            - p_t.clamp_min(eps).log()
            - p_s.clamp_min(eps).log()
        )
    ).sum()

    kl_t_s = (
        teacher_probs * (
            teacher_probs.clamp_min(eps).log()
            - student_probs.clamp_min(eps).log()
        )
    ).sum(dim=1).mean()

    kl_s_t = (
        student_probs * (
            student_probs.clamp_min(eps).log()
            - teacher_probs.clamp_min(eps).log()
        )
    ).sum(dim=1).mean()

    return {
        "I_A_T_A_S": I_A_T_A_S.item(),
        "KL_T_S": kl_t_s.item(),
        "KL_S_T": kl_s_t.item(),
        "joint": joint.detach().cpu(),
    }


def main(args):
    seed_everything(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    plot_dir = Path(args.outdir) / Path(args.exp_name) / "MI_plots"
    mat_dir = Path(args.outdir) / Path(args.exp_name) / "MI_matrices"
    ensure_dir(plot_dir)
    ensure_dir(mat_dir)
    metrics_history = {
        "teacher_H_A": [],
        "teacher_H_A_given_Z": [],
        "teacher_I_Z_A": [],
        "student_H_A": [],
        "student_H_A_given_Z": [],
        "student_I_Z_A": [],
        "teacher_student_I_A_T_A_S": [],
        "teacher_student_KL_T_S": [],
        "teacher_student_KL_S_T": [],
    }

    dataloader = get_dataloader()

    g = torch.Generator().manual_seed(args.seed)
    tokens_per_image = 197
    total_tokens = args.num_images * tokens_per_image
    indices = torch.randperm(total_tokens, generator=g)[:args.num_tokens]
    indices = indices.sort().values

    teacher = create_model(args.teacher_model, pretrained=False)
    ckpt = torch.load(args.teacher_path, map_location="cpu")
    msg = teacher.load_state_dict(ckpt, strict=False)
    print(msg)
    teacher.eval().to(device)

    t_mean, t_std = compute_bn_stats(model=teacher,
                                     dataloader=dataloader,
                                     n_tokens=args.bn_tokens,
                                     path="teacher_val_bn_stats.pt",
                                     device=device)
    z_t = sample_tokens(model=teacher,
                        dataloader=dataloader,
                        indices=indices,
                        path="teacher_tokens.pt",
                        device=device)

    z_t_bn = (z_t - t_mean) / (t_std + 1e-6)

    ckpt_paths = get_checkpoint_paths(args.ckpts)
    for epoch, ckpt_path in ckpt_paths:
        print(f"epoch {epoch} | {ckpt_path}")

        student, P = load_student_and_proto(student_model_name=args.student_model,
                                            ckpt_path=ckpt_path,
                                            device=device)

        s_mean, s_std = compute_bn_stats(model=student,
                                         dataloader=dataloader,
                                         n_tokens=args.bn_tokens,
                                         path=f"ckpt_e={epoch}_val_bn_stats.pt",
                                         device=device,
                                         use_projector=True)
        z_s = sample_tokens(model=student,
                            dataloader=dataloader,
                            indices=indices,
                            path=f"ckpt_e={epoch}_tokens.pt",
                            device=device,
                            use_projector=True)

        p_mean, p_std = compute_tensor_bn_stats(P,
                                                path=f"ckpt_e={epoch}_proto_bn_stats.pt")

        P_bn = (P - p_mean) / (p_std + 1e-6)
        z_s_bn = (z_s - s_mean) / (s_std + 1e-6)

        print_stats(z_t_bn, "Teacher BN Features")
        print_stats(z_s_bn, "Student BN Features")
        print_stats(P_bn, "Prototypes BN Features")

        teacher_probs, C_t = assignment_probs(z_t_bn.to(device), P_bn.to(device))
        student_probs, C_s = assignment_probs(z_s_bn.to(device), P_bn.to(device))

        teacher_metrics = entropy_metrics(teacher_probs)
        student_metrics = entropy_metrics(student_probs)
        teacher_student_metrics = teacher_student_assignment_metrics(teacher_probs, student_probs)

        metrics_history["teacher_H_A"].append(teacher_metrics["H_A"])
        metrics_history["teacher_H_A_given_Z"].append(teacher_metrics["H_A_given_Z"])
        metrics_history["teacher_I_Z_A"].append(teacher_metrics["I_Z_A"])

        metrics_history["student_H_A"].append(student_metrics["H_A"])
        metrics_history["student_H_A_given_Z"].append(student_metrics["H_A_given_Z"])
        metrics_history["student_I_Z_A"].append(student_metrics["I_Z_A"])

        metrics_history["teacher_student_I_A_T_A_S"].append(teacher_student_metrics["I_A_T_A_S"])
        metrics_history["teacher_student_KL_T_S"].append(teacher_student_metrics["KL_T_S"])
        metrics_history["teacher_student_KL_S_T"].append(teacher_student_metrics["KL_S_T"])

        print(f"\n[epoch {epoch}] metrics:")
        print(
            f"teacher: "
            f"H(A)={teacher_metrics['H_A']:.6e} | "
            f"H(A|Z)={teacher_metrics['H_A_given_Z']:.6e} | "
            f"I(Z;A)={teacher_metrics['I_Z_A']:.6e}"
        )
        print(
            f"student: "
            f"H(A)={student_metrics['H_A']:.6e} | "
            f"H(A|Z)={student_metrics['H_A_given_Z']:.6e} | "
            f"I(Z;A)={student_metrics['I_Z_A']:.6e}"
        )
        print(
            f"teacher-student: "
            f"I(A_T;A_S)={teacher_student_metrics['I_A_T_A_S']:.6e} | "
            f"KL(T||S)={teacher_student_metrics['KL_T_S']:.6e} | "
            f"KL(S||T)={teacher_student_metrics['KL_S_T']:.6e}"
        )

        matrices_to_save = {
            "teacher_C": C_t.detach().cpu(),
            "student_C": C_s.detach().cpu(),
            "teacher_probs": teacher_probs.detach().cpu(),
            "student_probs": student_probs.detach().cpu(),
            "teacher_student_joint": teacher_student_metrics["joint"],
        }

        for name, M in matrices_to_save.items():
            M_crop = M[:args.plot_crop_size, :args.plot_crop_size].detach().cpu()

            plt.figure()
            plt.imshow(M_crop.numpy())
            plt.colorbar()
            plt.title(f"{name} | ep={epoch}")
            plt.tight_layout()
            plt.savefig(mat_dir / f"ep={epoch}_{name}.png")
            plt.close()

    xs = [10 * x for x in range(1, len(metrics_history["teacher_H_A"]) + 1)]

    for name, vals in metrics_history.items():
        plt.figure()
        plt.plot(xs, vals)
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-model", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--teacher-path", default="/media/storage/katsikasd/KD_checkpoints/vit_small_patch16_dinov3.lvd1689m.bin")
    parser.add_argument("--student-model", default="vit_tiny_patch16_dinov3")

    parser.add_argument("--ckpts",
                        default="/media/storage/katsikasd/KD_checkpoints/vit_tiny/ablation_prot_train/detatch_dist_loss_0.5_tp_0.5_sp/s_vit_tiny_patch16_dinov3_t_vit_small_patch16_dinov3.lvd1689m_bs_1024_proj_matrix_normalize_True_d_KL_d_soft_cj_0.3_a_0.0_b_0.0_g_0.0_d_1.0_KoLeoD_0.0_KoLeoP_0.0_K_6304_sids_0511_tids_0511_prototypes_[0, 0, 3000]/")
    parser.add_argument("--outdir", default="./gaussian_kernel")
    parser.add_argument("--exp_name", default="detatch_dist_loss_0.5_tp_0.5_sp_on_stud")

    parser.add_argument("--kernel", default="gaussian")
    parser.add_argument("--sigma", default=0.1)

    parser.add_argument("--bn_tokens", type=int, default=600000, help='tokens to calculate bn stats')
    parser.add_argument("--num_tokens", type=int, default=40000, help='token pool')
    parser.add_argument("--num_images", type=int, default=15000, help='image pool')
    parser.add_argument("--plot_crop_size", type=int, default=1024)

    parser.add_argument("--pinv-rcond", type=float, default=1e-6)
    parser.add_argument("--nystrom-jitter", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--imagenet", default="/home/katdimitris55/Downloads/imagenet/val/")

    args = parser.parse_args()
    main(args)