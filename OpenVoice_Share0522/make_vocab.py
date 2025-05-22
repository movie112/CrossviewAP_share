"""
INPUT MFA 결과 TextGrid
OUTPUT VOCAB.JSON
"""

import os
import json
import tgt
from tqdm import tqdm
from collections import Counter, defaultdict
import re
import argparse
from konlpy.tag import Okt  

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/yeonghwa/workspace/flitto/CrossviewAP_Share0522/datasets')
parser.add_argument('--cv_dir', type=str, default='/home/yeonghwa/workspace/flitto/CrossviewAP_Share0522')
parser.add_argument('--lang', type=str, default='kr', help='kr, en, jp, zh')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--min_count',type=int, default=0, help='vocab의 단어 등장횟수 제한')
parser.add_argument("--morph", type=str, default="False", help="spliting per morph for Korean")
args = parser.parse_args()

morph_flag = args.morph.lower() in ("true", "1", "yes", "y")

lang = args.lang
mode = args.mode
data_dir = args.data_dir
min_count = args.min_count

tg_dir = f"{data_dir}/mfa/{mode}/{lang}"
if morph_flag:
    tg_dir = f"{data_dir}/mfa/{mode}/{lang}_morph"
if mode != "test":
    tg_dir = f"{data_dir}/mfa_{mode}/{lang}"
    if not morph_flag and lang == 'kr':
        tg_dir = f"{data_dir}/mfa_{mode}/{lang}_full"
subwords2ids_path = f"{args.cv_dir}/subwords_to_ids.json"
xsampa2ipa_path = f"{args.cv_dir}/xsampa_phonetic_features.json"
print(tg_dir)
out_path = f"{data_dir}/vocab/{mode}/{lang}/vocab_{min_count}.json"
if morph_flag:
    out_path = f"{data_dir}/vocab/{mode}/{lang}_morph/vocab_{min_count}.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

#  phones_to_ids 로드
#  원본 phones_to_ids 로드 (ID는 무시)
with open(subwords2ids_path, 'r', encoding='utf-8') as f:
    orig_phones2ids = json.load(f)
with open(xsampa2ipa_path, 'r', encoding='utf-8') as f:
    orig_xsampa = json.load(f)
# IPA → X-SAMPA 강제 매핑 테이블 정의
ipa2xs_map = {
    'zh':{  "aw˧": "aw", "m̩˧": "m", "ʐ̩˧": "Z", "u˧": "u", "i˧": "i", "ow˩": "ow","a˥˩": "a","spn": "<noise>","tɕ": "tS","o˨˩˦": "o","i˨˩˦": "i","ɕ": "S","u˥˩": "u","a˥": "a","ʈʂ": "tS","i˥": "i","o˥˩": "o","e˥˩": "e","i˥˩": "i","a˨˩˦": "a","tɕʰ": "tS_h","ej˥˩": "ej","ʐ̩˥˩": "Z","aw˥˩": "aw","ʈʂʰ": "tSh","aw˨˩˦": "aw","a˧˥": "a","o˧˥": "o","i˧˥": "i","ow˥˩": "ow","o˥": "o","e˨˩˦": "e","aj˥˩": "aj","e˧˥": "e","pʷ": "p","ə˨˩˦": "@","u˥": "u","ow˨˩˦": "ow","y˥˩": "y","ə˥": "@","e˥": "e","aj˧˥": "aj","ʐ̩˥": "Z","ə˧˥": "@","u˧˥": "u","tsʰ": "ts","ej˨˩˦": "ej","ej˥": "ej","ə˧": "@","ʐ̩˧˥": "Z","tʷ": "t","ow˥": "ow","ej˧˥": "ej","u˨˩˦": "u","ow˧˥": "ow","tɕʷ": "tS","ə˩": "@","z̩˥˩": "z","aj˨˩˦": "aj","aj˥": "aj","ə˥˩": "@","aw˥": "aw","y˨˩˦": "y","ʐ̩˨˩˦": "Z","aw˧˥": "aw","y˥": "y","ɕʷ": "S","y˧˥": "y","z̩˥": "z","z̩˧": "z","z̩˩": "z","ʐ̩˩": "Z","z̩˨˩˦": "z","o˩": "o","z̩˧˥": "z","aj˩": "aj","a˧": "a","e˩": "e","u˩": "u","i˩": "i","a˩": "a","o˧": "o","ej": "ej","ʐ̩": "Z","ow˧": "ow", "ow": "ow"},
    "en":{"ʈʷ": "t`","ɹ": "r","ɒ": "O","aj": "aj","spn": "<noise>","əw": "ow","dʒ": "dZ","ʉː": "u:","tʃ": "tS","d̪": "d","aw": "aw","ɒː": "O:","ɚ": "@","cʰ": "c","t̪": "t","ɜː": "@","ʈʲ": "tj","ʉ": "u","ej": "ej","ow": "ow","ɔj": "oj","ɝ": "@","cʷ": "c","tʷ": "t","ɟʷ": "J" },
    "jp":{"ɯ": "M\\","ɕ": "S","ɴ": "N","spn": "<noise>","ɾʲ": "4",        "dʑ": "dZ'","tɕ": "tS","ʑ": "Z'","ɸ": "p",         "tː": "t","ɨ̥": "I",         "ɯ̥": "M\\","i̥": "i","kː": "k","sː": "s","pː": "p","nː": "n","ɕː": "S","ɯː": "M\\","cː": "c","mː": "m","ɸʲ": "pj","tɕː": "tS","ɲː": "J","mʲː": "mj","dː": "d","ɡː": "g","pʲː": "pj","tsː": "ts","dʑː": "dZ'","dzː": "dz","tʲː": "tj","ɸː": "p","bː": "b"},
    "kr": {"spn": "<noise>","tɕ": "tS","sʰ": "s", "k̚": "k", "tɕʰ": "tS_h","dʑ": "dZ'","ɕʰ": "S",  "ɾʲ": "4",            "ʌː": "V",            "p̚": "p","ɸʷ": "p",             "ʎː": "L",             "t̚": "t","k͈": "k",             "βʷ": "v",             "tɕʷ": "tS",        "ɭː": "l`",          "tʷ": "t","t͈": "t","ɕ͈": "S","tɕ͈": "tS","nː": "n","cʰ": "c","p͈": "p","s͈": "s","tɕʷː": "tS","kʰː": "kh","pʷ": "p","dʑʷ": "dZ'","k͈ʷ": "k","t͈ʲ": "tj","tɕʰː": "tS_h","sʷ": "s","mː": "m","cʰː": "c","c͈": "c","tʰː": "th","pʰː": "ph","p͈ʲ": "pj","kʷː": "kw","bʷ": "b","pʲː": "pj","tɕ͈ː": "tS","ɾʷ": "4","mʲː": "mj","tɕː": "tS","sʰː": "s","tɕ͈ʷ": "tS"}
}

# phones 맵 가져오기
phones_map = orig_xsampa['phones']  # 각 키가 X-SAMPA, 값의 첫번째 원소가 IPA 
# IPA → X-SAMPA 매핑 생성
ipa_to_xsampa = { info[0]: xs for xs, info in phones_map.items() }

# 매핑 함수 정의
def ipa_to_xs(ipa_symbol):
    return ipa_to_xsampa.get(ipa_symbol)

# invalid 매핑 자동 제거
for ipa, xs in list(ipa2xs_map[lang].items()):
    if xs not in orig_phones2ids:
        ipa2xs_map[lang].pop(ipa)

# 4) TextGrid 파일 수집
def find_textgrids(root_dir):
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith('.TextGrid'):
                yield os.path.join(root, fn)

textgrid_paths = list(find_textgrids(tg_dir))
print(f"Found {len(textgrid_paths)} TextGrid files.")

# 5) TextGrid 파싱 함수
def parse_textgrid_word2phones(path):
    tg = tgt.read_textgrid(path)
    word_tier  = tg.get_tier_by_name('words')
    phone_tier = tg.get_tier_by_name('phones')
    out = []
    for w_iv in word_tier._objects:
        w = w_iv.text.strip()
        if not w:
            continue
        phs = [
            p.text.strip()
            for p in phone_tier._objects
            if (p.start_time >= w_iv.start_time
                and p.end_time <= w_iv.end_time
                and p.text.strip())
        ]
        out.append((w, phs))
    return out

# 6) 카운터 및 저장 구조 초기화
word_counter    = Counter()
phone_counter   = Counter()
char_counter    = Counter()
word_to_phones  = defaultdict(set)
unmapped_phones = Counter()

# 7) 전체 파일 순회 및 수집
for tg_path in tqdm(textgrid_paths, desc='Parsing TextGrids'):
    for word, phones in parse_textgrid_word2phones(tg_path):
        xs_phones = []
        for p in phones:
            xs = ipa_to_xs(p)
            if xs is None:
            # if p in orig_phones_to_ids:
            #     xs = p
                if p in ipa2xs_map[lang] and ipa2xs_map[lang][p] in orig_phones2ids:
                    xs = ipa2xs_map[lang][p]
                else:
                    unmapped_phones[p] += 1
                    continue
            xs_phones.append(xs)
            phone_counter[xs] += 1

        if not xs_phones:
            continue

        word_counter[word] += 1
        # 중복 제거: set에 tuple 저장
        word_to_phones[word].add(tuple(xs_phones))
        for ph in xs_phones:
            for c in ph:
                char_counter[c] += 1

# 8) set → list 변환
word_to_phones = {
    w: [list(seq) for seq in seqs]
    for w, seqs in word_to_phones.items()
}

# 9) 매핑 안된 phones 출력 및 저장
if unmapped_phones:
    print("Unmapped phones detected (phone: count):")
    for ph, cnt in unmapped_phones.most_common():
        print(f"{ph}: {cnt}")
    # with open('vunmapped_phones.json', 'w', encoding='utf-8') as umf:
    #     json.dump(dict(unmapped_phones), umf, ensure_ascii=False, indent=2)

def is_valid_word(word, lang):
    if lang == 'zh':  # 중국어
        return re.fullmatch(r'^[\u4e00-\u9fff]+$', word) is not None
    elif lang == 'kr':  # 한국어
        return re.fullmatch(r'^[\uac00-\ud7af]+$', word) is not None
    elif lang == 'jp':  # 일본어 (히라가나, 가타카나, CJK 통합 한자)
        return re.fullmatch(r'^[\u3040-\u30ff\u4e00-\u9fff]+$', word) is not None
    elif lang == 'en':  # 영어 (라틴 문자)
        return re.fullmatch(r'^[a-zA-Z]+$', word) is not None
    else:
        return False
# 10-1) 필터링된 word 목록 생성
valid_words = [
    w for w, cnt in word_counter.items()
    if cnt >= min_count
    and is_valid_word(w, lang)
    and not (len(word_to_phones.get(w, [])) == 1
             and word_to_phones[w][0] == ['<noise>'])
]

# 10-2) word_to_ids 생성
word_to_ids = {w: i for i, w in enumerate(sorted(valid_words))}

# 10-3) 사용된 phone 집합만 추출
used_phones = set()
for w in word_to_ids:
    for seq in word_to_phones[w]:
        used_phones.update(seq)

# 10-4) reduced_phones_to_ids: 0부터 시작
reduced_phones = sorted(used_phones)
phones_to_ids = {ph: idx for idx, ph in enumerate(reduced_phones)}

# 10-5) chars_to_ids 생성 (옵션)
chars_to_ids = {}
for phone in reduced_phones:
    if re.fullmatch(r"<.*?>", phone):
        chars_to_ids.setdefault(phone, len(chars_to_ids))
    else:
        for ch in phone:
            chars_to_ids.setdefault(ch, len(chars_to_ids))

# 10-6) word_to_phones / word_to_chars 를 valid_words 기준으로 재구성
filtered_word_to_phones = {
    w: word_to_phones[w]
    for w in word_to_ids
}
word_to_chars = {
    w: [[ph] if ph.startswith('<') else list(ph) 
        for seq in filtered_word_to_phones[w] for ph in seq]
    for w in word_to_ids
}

# 11) JSON 저장
vocab = {
    'words_to_ids':    word_to_ids,
    'phones_to_ids':  phones_to_ids,
    # 'chars_to_ids':   charSSSs_to_ids,
    'word_to_phones': filtered_word_to_phones,
    # 'word_to_chars':  word_to_chars,
}


with open(out_path, 'w', encoding='utf-8') as fout:
    json.dump(vocab, fout, ensure_ascii=True, indent=2)

print(f"{out_path} 생성 완료.")