import os
import os.path as osp
import h5py
import numpy as np
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import torch

from model import *
import vsum

metric = 'summe' #u can change it between tvsum and summe
dataset_path = f'datasets/eccv16_dataset_{metric}_google_pool5.h5'
save_dir = 'log'
split_count = 5

input_dim = 1024
hidden_dim = 256
num_layers = 2
rnn_cell = 'lstm'

lr = 1e-4
weight_decay = 1e-05
max_epoch = 50
stepsize = 30
gamma = 0.1
seed = 1

def main():
    os.makedirs(save_dir, exist_ok=True)
    print("Loading dataset {}".format(dataset_path))
    dataset = h5py.File(dataset_path, 'r')
    all_keys = sorted(list(dataset.keys()))
    print("Total videos:", len(all_keys))

    kf = KFold(n_splits=split_count, shuffle=True, random_state=seed)
    mean_fscores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_keys)):
        print("\n--- Fold {}/{} ---".format(fold + 1, split_count))
        train_keys = [all_keys[i] for i in train_idx]
        test_keys = [all_keys[i] for i in test_idx]
        print("# train videos {}, # test videos {}".format(len(train_keys), len(test_keys)))

        model = MLPScorer(in_dim=input_dim, hidden_dim=hidden_dim)
        print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
        criterion = nn.MSELoss()

        print("Training...")
        model.train()

        for epoch in range(max_epoch):
            np.random.shuffle(train_keys)
            epoch_losses = []

            for key in train_keys:
                seq = dataset[key]['features'][...]
                gtscore = dataset[key]['gtscore'][...]

                seq = torch.from_numpy(seq).unsqueeze(0)
                target = torch.from_numpy(gtscore).unsqueeze(0)
                pred = model(seq).squeeze()
                loss = criterion(pred, target.squeeze())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            print("Epoch {}/{} \tLoss: {:.6f}".format(epoch + 1, max_epoch, np.mean(epoch_losses)))
            scheduler.step()

        fscore = evaluate(model, dataset, test_keys, fold)
        mean_fscores.append(fscore)

        model_path = osp.join(save_dir, f'model_fold{fold + 1}.pth.tar')
        torch.save(model.state_dict(), model_path)
        print("Model saved to", model_path)

    dataset.close()
    print("\nCross-validation complete.")
    print("Fold-wise F-scores:", ["{:.2%}".format(f) for f in mean_fscores])
    print("Average F-score: {:.2%}".format(np.mean(mean_fscores)))

    results_path = osp.join(save_dir, "fold_results.txt")
    with open(results_path, "w") as f:
        f.write("F-score Results by Fold\n")
        f.write("========================\n")
        for i, fscore in enumerate(mean_fscores):
            f.write(f"Fold {i + 1}: ({fscore:.2%})\n")
        f.write("------------------------\n")
        avg_f = np.mean(mean_fscores)
        f.write(f"Average F-score: {avg_f:.4f} ({avg_f:.2%})\n")

def evaluate(model, dataset, test_keys, fold_idx):
    print("Evaluating Fold {}".format(fold_idx + 1))
    model.eval()
    fms = []
    eval_metric = 'avg' if metric == 'tvsum' else 'max'
    with torch.no_grad():
        for idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            preds = model(seq).data.squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            n_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            picks = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            model_summary = vsum.generate_summary(preds, cps, n_frames, nfps, picks)
            fm = vsum.evaluate_summary(model_summary, user_summary, eval_metric)
            fms.append(fm)
            print(f"{key}: "+'{:.2%}'.format(fm))

    mean_fm = np.mean(fms)
    print("Fold {} Average F-score: {:.2%}".format(fold_idx + 1, mean_fm))
    return mean_fm

if __name__ == '__main__':
    main()
