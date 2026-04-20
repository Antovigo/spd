"""Train a RedundantCopyTransformer on the copy task."""

import torch
import wandb
from torch import Tensor
from tqdm import tqdm, trange

from spd.configs import ScheduleConfig
from spd.experiments.completeness.configs import CompletenessModelConfig, CompletenessTrainConfig
from spd.experiments.completeness.models import CopyTaskDataset, RedundantCopyTransformer
from spd.log import logger
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import get_scheduled_value, set_seed
from spd.utils.run_utils import ExecutionStamp, save_file


def train(
    model: RedundantCopyTransformer,
    dataloader: DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    log_wandb: bool,
    steps: int,
    lr_schedule: ScheduleConfig,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr_schedule.start_val)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = get_scheduled_value(step, steps, lr_schedule)
            for group in opt.param_groups:
                group["lr"] = step_lr

            opt.zero_grad(set_to_none=True)
            tokens, targets = next(data_iter)
            logits = model(tokens)
            loss = loss_fn(logits, targets)
            loss.backward()
            opt.step()

            if step % 100 == 0 or step + 1 == steps:
                tqdm.write(f"Step {step} Loss: {loss.item():.4f}")
                t.set_postfix(loss=loss.item(), lr=step_lr)
                if log_wandb:
                    wandb.log({"loss": loss.item(), "lr": step_lr}, step=step)


def _make_ablation_configs(n_layers: int) -> list[tuple[str, list[bool]]]:
    configs: list[tuple[str, list[bool]]] = [("all active", [True] * n_layers)]
    for i in range(n_layers):
        mask = [False] * n_layers
        mask[i] = True
        configs.append((f"only layer {i}", mask))
    configs.append(("none active", [False] * n_layers))
    return configs


def _forward_with_active_layers(
    model: RedundantCopyTransformer,
    tokens: Tensor,
    active_mask: list[bool],
) -> Tensor:
    """Run the model applying only layers where `active_mask[i]` is True. Returns logits at pos 1."""
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    x = model.token_embed(tokens) + model.pos_embed(positions)
    for i, layer in enumerate(model.layers):
        if active_mask[i]:
            x = x + layer(x)
    return model.unembed(model.linear(x))[:, 1, :]


def verify_layer_ablations(
    model: RedundantCopyTransformer,
    dataset: CopyTaskDataset,
    n_samples: int = 5,
) -> None:
    """Print sample inputs and top-5 output tokens per ablation config."""
    model.eval()
    tokens, targets = dataset.generate_batch(n_samples)

    logger.info(f"Sample inputs (token 0 = value, token 1 = eq_token={dataset.eq_token}):")
    for j in range(n_samples):
        logger.info(f"  example {j}: tokens={tokens[j].tolist()}, target={targets[j].item()}")

    with torch.no_grad():
        for name, active_mask in _make_ablation_configs(len(model.layers)):
            probs = _forward_with_active_layers(model, tokens, active_mask).softmax(dim=-1)
            logger.info(f"\n  {name}:")
            for j in range(n_samples):
                top_probs, top_ids = probs[j].topk(5)
                entries = [
                    f"{tid.item()}({tp:.3f})" for tid, tp in zip(top_ids, top_probs, strict=True)
                ]
                correct = "Y" if top_ids[0].item() == targets[j].item() else "N"
                logger.info(
                    f"    example {j} [target={targets[j].item()}]: {' '.join(entries)}  {correct}"
                )


def verify_bulk_accuracy(model: RedundantCopyTransformer, dataset: CopyTaskDataset) -> None:
    model.eval()
    tokens, targets = dataset.generate_batch(4096)
    with torch.no_grad():
        for name, active_mask in _make_ablation_configs(len(model.layers)):
            logits = _forward_with_active_layers(model, tokens, active_mask)
            acc = (logits.argmax(dim=-1) == targets).float().mean().item()
            logger.info(f"  {name}: accuracy = {acc:.4f}")


def run_train(config: CompletenessTrainConfig, device: str | torch.device) -> None:
    model = RedundantCopyTransformer(config.completeness_model_config)
    model.layer_dropout_p = config.layer_dropout_p
    model.to(device)
    model.train()

    dataset = CopyTaskDataset(
        vocab_size=config.completeness_model_config.vocab_size,
        eq_token=config.completeness_model_config.eq_token,
        device=device,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    mc = config.completeness_model_config
    run_name = f"completeness_v{mc.vocab_size}_d{mc.d_model}_L{mc.n_layers}_seed{config.seed}"

    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        wandb.init(
            id=execution_stamp.run_id,
            project=config.wandb_project,
            name=run_name,
            tags=["completeness"],
        )

    config_path = out_dir / "completeness_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    train(
        model,
        dataloader=dataloader,
        log_wandb=config.wandb_project is not None,
        steps=config.steps,
        lr_schedule=config.lr_schedule,
    )

    model_path = out_dir / "completeness.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    logger.info("Layer ablation verification (sample outputs):")
    verify_layer_ablations(model, dataset)

    logger.info("\nBulk accuracy per ablation:")
    verify_bulk_accuracy(model, dataset)


if __name__ == "__main__":
    device = get_device()

    config = CompletenessTrainConfig(
        wandb_project="spd",
        completeness_model_config=CompletenessModelConfig(
            vocab_size=16,
            d_model=64,
            n_layers=2,
            seq_len=2,
            eq_token=0,
        ),
        layer_dropout_p=0.4,
        batch_size=1024,
        steps=5000,
        seed=0,
        lr_schedule=ScheduleConfig(start_val=1e-3, fn_type="cosine", final_val_frac=0.0),
    )

    set_seed(config.seed)
    run_train(config, device)
