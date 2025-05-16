import numpy as np

# 파일 경로
path_label = "C:\\Users\\movie\\Downloads\\label_61-70968-0000_8455-210777-0012.npy"
path_attractor = "C:\\Users\\movie\\Downloads\\attractor_61-70968-0000_8455-210777-0012.npy"
path_emb = "C:\\Users\\movie\\Downloads\\attractor_KRSMD2I094.npy"

# 파일 불러오기
arr_label = np.load(path_label)
arr_attractor = np.load(path_attractor)
arr_emb = np.load(path_emb)

# shape(차원) 확인
print("arr_label shape:", arr_label.shape)
print("arr_attractor shape:", arr_attractor.shape)
print("arr_emb shape:", arr_emb.shape)

# 실제 데이터 일부 확인 (처음 5개 정도)
print("\narr_label[:5]:\n", arr_label[:5])
print("\narr_attractor[:5]:\n", arr_attractor[:5])
print("\narr_emb[:5]:\n", arr_emb[:5])
