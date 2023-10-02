
import string
import torch
from strhub.models.parclip.CLIP import clip, simple_tokenizer
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def kmeans(data, n_clusters, max_iters=2000):
    n_samples, n_features = data.size()
    centroids = data[torch.randperm(n_samples)[:n_clusters]]

    for _ in range(max_iters):
        # 각 데이터 포인트에 가장 가까운 센트로이드 찾기
        distances = torch.norm(data.unsqueeze(1) - centroids.unsqueeze(0), dim=-1)
        labels = torch.argmin(distances, dim=1)

        # 새로운 센트로이드 계산
        new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(n_clusters)])

        # 센트로이드가 변하지 않을 때 종료
        if torch.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return labels, centroids

def dbscan(data):
    
    import torch
    from sklearn.cluster import DBSCAN

    # DBSCAN 클러스터링 수행
    dbscan = DBSCAN(eps=0.3, min_samples=10)

    data = data.cpu().detach().numpy()
    # 텐서를 NumPy 배열로 변환하지 않고 바로 텐서를 사용하여 클러스터링
    labels = dbscan.fit_predict(data)
    return labels


def main():
    CLIPmodel, CLIPpreprocess = clip.load('ViT-B/16')
    charset_train = string.digits + string.ascii_lowercase

    dic = simple_tokenizer.SimpleTokenizer(max_label_length= 25, charset = charset_train)
    label = dic.getLabelVocab()

    label = label[:200]

    print(len(label))
    text_token = torch.cat([clip.tokenize(f"word {c}") for c in label]).to(device)
    text_features = CLIPmodel.encode_text(text_token.to(device))
    print(text_features.shape)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    data = text_features

    # K-means 클러스터링 수행
    n_clusters = 10
    #labels, centroids = kmeans(data, n_clusters)
    labels = dbscan(data)

    data = data.cpu().detach().numpy()
    #labels = labels.cpu()
    #centroids = centroids.cpu().detach().numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
    for x,y,l in zip(data[:,0], data[:,1], label):
        plt.text(x, y, l, fontsize=6, ha='center', va='bottom')
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.legend()
    plt.title("K-means Clustering with PyTorch Tensor")
    plt.show()
    plt.savefig("kmeans_result.png",  bbox_inches='tight', dpi=1000)
    print("done")


def calculate_overlap_ratio(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            content1 = set(file1.read().splitlines())
            content2 = set(file2.read().splitlines())
    except FileNotFoundError:
        return 0.0  # 파일을 찾을 수 없는 경우 겹치는 비율은 0

    # intersection = content1.intersection(content2)
    # overlap_ratio = len(intersection) / min(len(content1), len(content2))

    intersection_count = sum(1 for word in content2 if word in content1)
    overlap_ratio = intersection_count / len(content2)

    return overlap_ratio

if __name__ == '__main__':
    # 예시 사용
    file2_path = "/home/ohh/PycharmProject/PARCLIP/test_GT.txt"
    file1_path = "/home/ohh/PycharmProject/PARCLIP/labels.txt"
    overlap_ratio = calculate_overlap_ratio(file1_path, file2_path)
    print("Overlap Ratio:", overlap_ratio)