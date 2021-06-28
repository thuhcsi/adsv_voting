from argparse import ArgumentParser
import torch
from scipy.io import wavfile

from torch_speaker.utils import cfg, load_config
from torch_speaker.module import Task
from torch.score import compute_eer

def load_wav(path):
    sample_rate, audio = wavfile.read(path)
    audio = torch.FloatTensor(audio)
    if self.device == "cuda":
        audio = audio.cuda()
    return sample_rate, audio

class Adversarial_Attack_Helper(object):
    def __init__(self, model, alpha=3.0, restarts=1, num_iters=5, epsilon=15, 
            adv_save_dir="data/adv_data/", device="cuda"):
        self.model = model
        self.alpha = alpha
        self.restarts = restarts
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.adv_save_dir = adv_save_dir
        self.adv_trials_path = os.path.join(adv_save_dir, "adv_trials.lst")
        self.device = device

        if not os.path.exists(os.path.join(adv_save_dir, "wav")):
            os.makedirs(os.path.join(adv_save_dir, "wav"))

        self.model.eval()
        if self.device == "cuda":
            self.model.cuda()

    def attack(self):
        # adversarial attack example generation
        adv_trials_file = open(self.adv_trials_path, "w")
        labels = []
        scores = []
        for idx, item in enumerate(tqdm(self.trials)):
            label, enroll_path, adv_test_path, score = self.bim_attack_step(idx, item)
            adv_trials_file.write("{} {} {}\n".format(label, enroll_path, adv_test_path))
            labels.append(int(label))
            scores.append(score)
        eer, th = compute_eer(labels, scores)
        print("EER: {:.3f} %".format(eer*100))


    def bim_attack_step(self, idx, item):
        label, enroll_path, test_path = item
        samplerate, enroll_wav = load_wav(enroll_path)
        samplerate, test_wav = load_wav(test_path)
        max_delta = torch.zeros_like(test_wav).cuda()

        # init best_score and alpha
        label = int(label)
        if label == 1:
            best_score = torch.tensor(float('inf')).cuda()
            alpha = self.alpha*(-1.0)
        else:
            best_score = torch.tensor(float('-inf')).cuda()
            alpha = self.alpha*(1.0)

        # extract enroll speaker embedding
        enroll_embedding = self.model.extract_speaker_embedding(enroll_wav).squeeze(0)

        for i in range(self.restarts):
            delta = torch.zeros_like(test_wav, requires_grad=True).cuda()
            for t in range(self.num_iters):
                # extract test speaker embedding
                input_wav = test_wav+delta
                test_embedding = self.model.extract_speaker_embedding(input_wav).squeeze(0)
                # cosine score
                score = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                score = score/denom

                # compute grad and update delta
                score.backward(retain_graph=True)
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-1*self.epsilon, self.epsilon)
                delta.grad.zero_()

            test_embedding = self.model.extract_speaker_embedding(test_wav+delta).squeeze(0)
            final_score = enroll_embedding.dot(test_embedding.T)
            denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
            final_score = final_score/denom

            if label == 1 and best_score >= final_score:
                max_delta = delta.data
                best_score = torch.min(best_score, final_score)
            elif label == 0 and best_score <= final_score:
                max_delta = delta.data
                best_score = torch.max(best_score, final_score)

        # Get Adversarial Attack wav
        adv_wav = test_wav + max_delta
        adv_wav = adv_wav.cpu().detach().numpy()
        final_score = final_score.cpu().detach().numpy()

        # save attack test wav
        idx = '%08d' % idx
        adv_test_path = os.path.join(self.adv_save_dir, "wav", idx+".wav")
        wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
        return label, enroll_path, adv_test_path, final_score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', help='train config file path', default="config/config.yaml")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path', default=None)
    parser.add_argument('--trial_path', type=str, help='trial file path', default=None)

    parser.add_argument('--alpha', help='', type=float, default=3.0)
    parser.add_argument('--restarts', help='', type=int, default=1)
    parser.add_argument('--num_iters', help='', type=int, default=5)
    parser.add_argument('--epsilon', help='', type=int, default=15)
    parser.add_argument('--adv_save_dir', help='', type=str, default="data/adv_data/")
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

    helper = Adversarial_Attack_Helper(model, args.alpha, args.restarts, 
            args.num_iters, args.epsilon, args.adv_save_dir, args.device)
    helper.attack()


