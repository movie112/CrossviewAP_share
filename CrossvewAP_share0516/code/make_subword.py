from g2pk import G2p
from jamo import h2j
import json

g2p = G2p()

# 한국어 vocab.json 경로
vocab_path = "/nas_homes/yeonghwa/flitto/datasets/ap_data_f/vocab/kr_full_vocab_6.json"  # 또는 사용 중인 한국어 vocab 파일명

# 단어들 불러오기
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

words = list(vocab["words_to_ids"].keys())

# 자모 수집
subword_set = set()
for word in words:
    try:
        pronounced = g2p(word)
        subwords = list(h2j(pronounced))
        subword_set.update(subwords)
    except:
        continue

# ID 할당 및 저장
subwords_to_ids = {sw: i for i, sw in enumerate(sorted(subword_set))}
with open("/home/yeonghwa/workspace/flitto/Multilingual-AWE_yh0513/kr_subwords_to_ids.json", "w", encoding="utf-8") as f:
    json.dump(subwords_to_ids, f, ensure_ascii=False, indent=2)

print(f"자모 수: {len(subwords_to_ids)}개 저장됨.")
