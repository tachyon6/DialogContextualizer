import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    tokenizer = AutoTokenizer.from_pretrained('bespin-global/klue-sroberta-base-continue-learning-by-mnr')
    model = AutoModel.from_pretrained('bespin-global/klue-sroberta-base-continue-learning-by-mnr')
    print("모델 및 토크나이저 로드 성공")
except Exception as e:
    print("모델 로드 중 오류 발생:", e)
    exit()

# 사전 정의된 상황과 그 임베딩
predefined_situations = [
    "호감",
    "사랑",
    "비호감",
    "혐오",
    "증오",
    "불쾌",
    "슬픔",
    "분노",
    "행복"
]

def generate_predefined_embeddings(situations):
    embeddings = []
    for situation in situations:
        inputs = tokenizer(situation, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(embeddings)

try:
    predefined_embeddings = generate_predefined_embeddings(predefined_situations)
    print("사전 정의된 상황의 임베딩 생성 성공")
except Exception as e:
    print("사전 정의된 상황 임베딩 생성 중 오류 발생:", e)
    exit()

def generate_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    except Exception as e:
        print("임베딩 생성 중 오류 발생:", e)
        return None

def generate_max_embedding(texts):
    embeddings = []
    for text in texts:
        embedding = generate_embedding(text)
        if embedding is not None:
            embeddings.append(embedding)
    if not embeddings:
        return None
    return np.max(embeddings, axis=0)

def find_most_similar_situation(texts):
    try:
        input_embedding = generate_max_embedding(texts)
        if input_embedding is None:
            return "임베딩 생성 실패"
        similarities = cosine_similarity(input_embedding.reshape(1, -1), predefined_embeddings)
        most_similar_index = np.argmax(similarities)
        return predefined_situations[most_similar_index]
    except Exception as e:
        print("가장 유사한 상황 찾기 중 오류 발생:", e)
        return "오류 발생"

def main():
    try:
        print("여러 문장을 입력하세요. 입력이 끝나면 빈 줄을 입력하세요.")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        
        if not lines:
            print("입력된 문장이 없습니다.")
            return
        
        most_similar_situation = find_most_similar_situation(lines)
        print("가장 유사한 상황: ", most_similar_situation)
    except Exception as e:
        print("입력 처리 중 오류 발생:", e)

if __name__ == "__main__":
    main()
