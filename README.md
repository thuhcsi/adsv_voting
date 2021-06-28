# Voting for the right answer

<!-- * [audio samples](https://zyzisyz.github.io/voting_audio_samples/)
* [arxiv paper](https://arxiv.org/abs/2106.07868) -->
## Framework
- As shown in the above figure, denoted as the conventional ASV system workflow, we get the ASV score from a piece of single utterance.
- However, such conventional ASV is highly vulnerable to adversarial samples (you can refer to the [adversarial audio samples](https://zyzisyz.github.io/voting_audio_samples/)) which are very similar to their original counterparts from human's perception, yet will manipulate the ASV render wrong prediction.
- In order to improve the adversarial robustness of such conventional ASV system, we propose the idea of "voting" to prevent risky decisions of ASV in blind spot areas, by employing random sampling neibours around the testing utterance and letting them vote for the right answer.
- [arxiv paper](https://arxiv.org/abs/2106.07868)
![](docs/workflow.png)


## Installation

```bash
git clone https://github.com/thuhcsi/adsv_voting
cd adsv_voting
git clone https://github.com/thuhcsi/torch_speaker
cd torch_speaker
pip install -r requirements.txt
python setup.py develop
cd ..
cp -r torch_speaker/{tools,scripts} .
```

## Experiments

**stage 1:** data preparation

```bash
rm -rf data; mkdir data
wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
echo format trails
python3 scripts/format_trials.py \
			--voxceleb1_root $voxceleb1_path \
			--src_trials_path data/veri_test.txt \
			--dst_trials_path data/vox1.txt
```

**stage 2:** ASV model evaluation in raw audio

```bash
python3 tools/evaluate.py \
			--config config/voting.yaml \
			--trial_path data/vox1.txt \
			--checkpoint_path $checkpoint_path
```

**stage 3:** adversarial attack and examples generation

```bash
python3 local/attack.py \
			--config config/voting.yaml \
			--trial_path data/vox1.txt \
			--checkpoint_path $checkpoint_path
```

**stage 4:** ASV model evaluation in adversarial examples

```bash
python3 tools/evaluate.py \
			--config config/voting.yaml \
			--trial_path data/vox1.txt \
			--checkpoint_path $checkpoint_path
```

**stage 5:** voting for the defense

```bash
python3 local/defense.py \
			--config config/voting.yaml \
			--trial_path data/vox1.txt \
			--checkpoint_path $checkpoint_path
```

## Citation

Please cite our paper if you make use of the code.

```
@article{wu2021voting,
	title={Voting for the right answer: Adversarial defense for speaker verification},
	author={Wu, Haibin and Zhang, Yang and Wu, Zhiyong and Wang, Dong and Lee, Hung-yi},
	booktitle={Interspeech},
	year={2021}
}
```
