import numpy as np
from knapsack import knapsack_dp
def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15):
    if positions.dtype != int:
        positions = positions.astype(int)
    if positions[-1] != n_frames:
        positions = np.append(positions, n_frames)

    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(positions) - 1):
        score = ypred[i] if i < len(ypred) else 0
        frame_scores[positions[i]:positions[i+1]] = score

    seg_scores = [
        frame_scores[int(start):int(end)+1].mean()
        for start, end in cps
    ]
    limit = int(n_frames * proportion)
    picks = knapsack_dp(seg_scores, nfps, len(nfps), limit)

    # Generate binary summary
    summary = np.concatenate([
        np.ones(nf, dtype=np.float32) if i in picks else np.zeros(nf, dtype=np.float32)
        for i, nf in enumerate(nfps)
    ])
    return summary[:n_frames]

def evaluate_summary(model_summary, user_summary, eval_metric='avg'):
    model_summary = (model_summary > 0).astype(np.float32)
    user_summary = (user_summary > 0).astype(np.float32)

    n_users, n_frames = user_summary.shape
    if len(model_summary) != n_frames:
        model_summary = np.pad(model_summary, (0, n_frames - len(model_summary)), mode='constant')[:n_frames]

    precisions = []
    recalls = []
    f_scores = []

    for gt in user_summary:
        overlap = (model_summary * gt).sum()
        precision = overlap / (model_summary.sum() + 1e-8)
        recall = overlap / (gt.sum() + 1e-8)
        f_score = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)

    if eval_metric == 'avg':
        return np.mean(f_scores)
    elif eval_metric == 'max':
        max_idx = np.argmax(f_scores)
        return f_scores[max_idx]
