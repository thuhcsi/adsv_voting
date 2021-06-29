import os
import argparse
import torch
import numpy as np

from torch_speaker.utils import cfg, load_config
from torch_speaker.module import Task

def load_wav(path, device="cuda"):
    sample_rate, audio = wavfile.read(path)
    audio = torch.FloatTensor(audio)
    if device == "cuda":
        audio = audio.cuda()
    return sample_rate, audio.unsqueeze(0)

def gaussian_voting_defense(trials, model, epsilon, num_voting=10, score_save_path="score.txt", device="cuda"):
    res = []
    for idx, item in enumerate(tqdm(trials)):
        label, enroll_path, test_path = item
        # load wav
        sample_rate, enroll_wav = load_wav(enroll_path)
        sample_rate, test_wav = load_wav(test_path)

        # init result list for this trials item
        res_item = []
        res_item.append(label)
        if device == "cuda":
            model = model.cuda()
            enroll_wav = enroll_wav.cuda()
            test_wav = test_wav.cuda()

        model.eval()
        with torch.no_grad():
            enroll_embedding = model.extract_embedding(enroll_wav).squeeze(0)
            test_embedding = model.extract_embedding(test_wav).squeeze(0)

            # score without guassion noise
            score = enroll_embedding.dot(test_embedding.T)
            denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
            score = score/denom
            score = score.cpu().detach().numpy().tolist()
            res_item.append(score)

            # score with guassion noise
            test_wav = test_wav.unsqueeze(0).repeat(num_voting, 1)
            gaussian_nosie = torch.randn_like(test_wav) * epsilon

            test_embeddings = model.extract_embedding(test_wav+gaussian_nosie)
            enroll_embedding = enroll_embedding.unsqueeze(0)
            scores = enroll_embedding.mm(test_embeddings.T)
            denom = torch.norm(enroll_embedding) * torch.norm(test_embeddings, dim=1)
            scores = scores/denom
            scores = scores.cpu().detach().numpy().tolist()[0]
            for score in scores:
                res_item.append(score)
        res.append(res_item)

    if score_save_path is not None:
        with open(score_save_path, "w") as f:
            for idx, item in enumerate(res):
                for val in item:
                    f.write(str(val) + " ")
                        f.write("\n")
    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', default="config/config.yaml")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path', default=None)
    parser.add_argument('--trial_path', type=str, help='trial file path', default=None)

    parser.add_argument('--epsilon', help='', type=int, default=5)
    parser.add_argument('--num_voting', help='', type=int, default=10)
    parser.add_argument('--device', help='', type=str, default="cuda")

    args = parser.parse_args()
    load_config(cfg, args.config)
    cfg.trainer.gpus = 1
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    if args.trial_path is not None:
        cfg.trial_path = args.trial_path

    model = Task(**cfg)
    if cfg.checkpoint_path is not None:
        state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")["state_dict"]
        # pop loss Function parameter
        loss_weights = []
        if cfg.keep_loss_weight is False:
            for key, value in state_dict.items():
                if "loss" in key:
                    loss_weights.append(key)
            for item in loss_weights:
                state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)
        print("initial parameter from pretrain model {}".format(cfg.checkpoint_path))
        print("keep_loss_weight {}".format(cfg.keep_loss_weight))

    trials = np.loadtxt(args.trial_path, str)
    gaussian_voting_defense(trials, model, args.epsilon, args.num_voting, args.score_save_path)


