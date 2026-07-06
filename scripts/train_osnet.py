import argparse
import torch
import torchreid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/person_reid_osnet/data")
    parser.add_argument("--save-dir", default="/root/autodl-tmp/person_reid_osnet/results/osnet_train")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()

    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=args.batch_size,
        batch_size_test=100,
        transforms=["random_flip", "random_erase"],
    )

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=10,
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    engine.run(
        save_dir=args.save_dir,
        max_epoch=args.epochs,
        eval_freq=5,
        print_freq=20,
        test_only=False,
    )

if __name__ == "__main__":
    main()
