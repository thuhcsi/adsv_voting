# voting for the right answer

* [audio samples](https://zyzisyz.github.io/voting_audio_samples/)
* [arxiv paper](https://arxiv.org/abs/2106.07868)

## installation

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

