import argparse
import torch
import torchreid

def load_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    fixed = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        fixed[k] = v
    model.load_state_dict(fixed, strict=False)
    print(f"Loaded weights: {weight_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/person_reid_osnet/data")
    parser.add_argument("--weights", default="/root/autodl-tmp/person_reid_osnet/results/osnet_train_60epoch/model/model.pth.tar-60")
    parser.add_argument("--save-dir", default="/root/autodl-tmp/person_reid_osnet/results/osnet_eval")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=64,
        batch_size_test=args.batch_size,
        transforms=["random_flip"],
    )

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False,
    )

    load_checkpoint(model, args.weights)

    if torch.cuda.is_available():
        model = model.cuda()

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=None,
        scheduler=None,
    )

    engine.run(
        save_dir=args.save_dir,
        max_epoch=0,
        eval_freq=1,
        print_freq=10,
        test_only=True,
    )

if __name__ == "__main__":
    main()
