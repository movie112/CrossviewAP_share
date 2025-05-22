# CrossviewAP_share

## Data Generation
### OpenVoice_Share0522
**1. make_audio.py**
- 참고
  - https://github.com/myshell-ai/OpenVoice.git
  - https://github.com/myshell-ai/MeloTTS
- Setting
  - OpenVoice_Share0522/checkpoints_v2 로 다운로드
    - https://drive.google.com/file/d/1G4LO-B-HGojpVMgB367S4nufSXH_jJuv/view?usp=drive_link
```
conda create -n openvoice python=3.9
conda activate openvoice

cd OpenVoice
pip install -e .
sudo apt install ffmpeg or conda install -c conda-forge ffmpeg

cd MeloTTS_w_noise
pip install -e .
pip install huggingface_hub==0.14.0 
```

**2. make_mfa.py**
- 참고: https://montreal-forced-aligner.readthedocs.io/en/v2.1.7/index.html
- 데이터 생성 시 누락 이슈 존재 -> 확인 후 해당 디렉토리만 재생성 작업
- Setting
  ```
  conda create -n mfa_env -c conda-forge montreal-forced-aligner
  conda activate mfa_env
  pip install joblib==1.2.0
  
  pip install python-mecab-ko jamo
  pip install spacy-pkuseg dragonmapper hanziconv
  conda install -c conda-forge spacy sudachipy sudachidict-core
  ```
**3-1. make_hdf5.py**
- `count_hdf.py`로 개수 체크 : 언어별 500
```
conda activate openvoice
```

**3-2. make_vocab.py**
```
conda activate openvoice
```
## Crossview AP Evaluation
### CrossviewAP_share0516
**4.evaluate.py**
- 참고: https://github.com/Yushi-Hu/Multilingual-AWE
- Setting
  - python 3.7, pytorch 1.3, h5py 2.9.0, numpy, scipy, tensorboard 1.14.0
  - CrossviewAP_share0516/ckpts, CrossviewAP_share0516/datasets 로 다운로드
    - https://drive.google.com/file/d/1oyoFQ5C-rqac2lEbYx_0dqMpUPRLsO-b/view?usp=sharing
    - https://drive.google.com/file/d/1pFck6ozBtjLcNdFeU4kfJgQY5xJTVGxY/view?usp=sharing
  - config 파일 내 경로 수정 필요
  
- result_dir 에 crossview ap 결과 저장
- kr_morph: 한국어 명사만 수집하여 평가
```
python evaluate.py --config {config path: CrossviewAP_Share0522/expt/sample} --result_dir {metric 저장 dir: CrossviewAP_Share0522/datasets/result}
OR
python evaluate_all.py # 파일 내 경로 수정
```
