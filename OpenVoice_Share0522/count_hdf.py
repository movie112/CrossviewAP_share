import h5py

lang = 'kr'
root_dir = '/home/yeonghwa/workspace/flitto/CrossviewAP_Share0522/datasets/hdf5/test'
# 파일 경로 설정
file_path = f"{root_dir}/{lang}/align.hdf5"

# HDF5 파일 열기
with h5py.File(file_path, 'r') as hf:
    total_count = 0
    for group_key in hf.keys():
        group = hf[group_key]
        if isinstance(group, h5py.Group):
            count = len(group)
            # print(f"{group_key}: {count} items")
            total_count += count
        else:
            # print(f"{group_key}: 1 item")
            total_count += 1

    print(f"총 데이터 개수: {total_count}")
    
file_path = f"{root_dir}/{lang}/align.hdf5"

# HDF5 파일 열기
with h5py.File(file_path, 'r') as hf:
    total_count = 0
    for group_key in hf.keys():
        group = hf[group_key]
        if isinstance(group, h5py.Group):
            count = len(group)
            # print(f"{group_key}: {count} items")
            total_count += count
        else:
            # print(f"{group_key}: 1 item")
            total_count += 1

    print(f"총 데이터 개수: {total_count}")
