from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import spacy
from language_tool_python import LanguageTool
from transformers import pipeline
nltk.download('punkt')

app = Flask(__name__)
CORS(app)
@app.route('/check_grammar', methods=['POST'])
def check_grammar():
    data = request.get_json()
    user_input = data['user_input']
    correct_answer = data['correct_answer']
    user_id = data.get('user_id', 'default_user')

    # Kiểm tra lỗi bằng so sánh trực tiếp
    has_error = user_input != correct_answer
    if not has_error:
        return jsonify({"error": False, "message": "Câu đúng!"})

    # Phân tích lỗi sai bằng spaCy
    user_doc = nlp(user_input)
    correct_doc = nlp(correct_answer)
    errors = []
    for i, (user_token, correct_token) in enumerate(zip(user_doc, correct_doc)):
        if user_token.text != correct_token.text:
            if user_token.pos_ == "VERB":
                errors.append(f"Lỗi thì động từ: '{user_token.text}' nên là '{correct_token.text}' (thì {correct_token.tag_}).")
            elif user_token.dep_ == "prep":
                errors.append(f"Lỗi giới từ: '{user_token.text}' nên là '{correct_token.text}'.")

    # Lưu lỗi vào lịch sử
    if user_id not in user_error_history:
        user_error_history[user_id] = []
    user_error_history[user_id].append({"type": "grammar", "error": errors})

    # Xác định cấp độ người dùng
    user_errors = user_error_history[user_id]
    error_count = len(user_errors)
    level = "Người mới bắt đầu" if error_count > 5 else "Trung cấp" if error_count > 2 else "Nâng cao"

    # Gợi ý tài liệu
    suggestion = "Học 'English Grammar in Use' (Raymond Murphy) về thì động từ." if "VERB" in [e.pos_ for e in user_doc] else "Xem bài học giới từ trên BBC Learning English."
    if level == "Người mới bắt đầu":
        suggestion += " Bắt đầu với Duolingo."

    return jsonify({
        "error": True,
        "errors": errors,
        "corrected": correct_answer,
        "suggestion": suggestion,
        "level": level
    })

# Tải mô hình spaCy và BERT
nlp = spacy.load("en_core_web_sm")
semantic_analyzer = pipeline("text-classification", model="bert-base-uncased")

# Tải dữ liệu huấn luyện và huấn luyện Word2Vec
with open("data/training_data.txt", "r", encoding="utf-8") as f:
    text_data = f.readlines()
sentences = [sent_tokenize(text.strip()) for text in text_data if text.strip()]
sentences = [sentence for sublist in sentences for sentence in sublist]
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
word2vec_model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Hàm chuyển câu thành vector
def sentence_to_vector(sentence):
    words = word_tokenize(sentence.lower())
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

# Huấn luyện Isolation Forest và K-Means cho phần viết
# Huấn luyện Isolation Forest và K-Means
writing_vectors = [sentence_to_vector(sentence) for sentence in sentences]
writing_vectors = np.array(writing_vectors, dtype=np.float64)  # Đã có dòng này, nhưng kiểm tra lại
print("writing_vectors dtype:", writing_vectors.dtype)  # Debug kiểu dữ liệu
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(writing_vectors)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(writing_vectors)

# Khởi tạo LanguageTool
tool = LanguageTool('en-US')

# Lưu trữ lịch sử lỗi
user_error_history = {}

@app.route('/check_writing', methods=['POST'])
def check_writing():
    data = request.get_json()
    user_input = data['user_input']
    correct_sentence = data['correct_sentence']
    user_id = data.get('user_id', 'default_user')

    # Phát hiện lỗi bằng Isolation Forest
    user_vector = sentence_to_vector(user_input)
    print("user_vector dtype:", user_vector.dtype)
    print("user_vector shape:", user_vector.shape)
    user_vector = np.array(user_vector, dtype=np.float64)
    prediction = iso_forest.predict([user_vector])[0]
    has_error = prediction == -1

    if not has_error:
        return jsonify({"error": False, "message": "Câu đúng!"})

    # Phân cụm lỗi bằng K-Means
    print("Before KMeans predict - user_vector dtype:", user_vector.dtype)
    cluster = kmeans.predict([user_vector])[0]
    error_type = "Ngữ pháp" if cluster == 0 else "Từ vựng"

    # Phân tích lỗi bằng spaCy
    user_doc = nlp(user_input)
    correct_doc = nlp(correct_sentence)
    errors = []
    for i, (user_token, correct_token) in enumerate(zip(user_doc, correct_doc)):
        if user_token.text != correct_token.text:
            if user_token.pos_ == "VERB":
                errors.append(f"Lỗi chia động từ: '{user_token.text}' nên là '{correct_token.text}' (thì {correct_token.tag_}).")
            else:
                errors.append(f"Lỗi từ vựng: '{user_token.text}' nên là '{correct_token.text}'.")

    # Phân tích ngữ nghĩa bằng BERT
    semantic_check = semantic_analyzer(user_input)
    semantic_error = "Có thể có lỗi ngữ nghĩa, hãy kiểm tra ngữ cảnh." if semantic_check[0]['score'] < 0.7 else None
    if semantic_error:
        errors.append(semantic_error)

    # Đề xuất chỉnh sửa
    corrected = tool.correct(user_input)

    # Nghiên cứu nguyên nhân
    reason = "Người dùng có thể không nắm vững quy tắc chia động từ." if error_type == "Ngữ pháp" else "Người dùng có thể thiếu vốn từ vựng hoặc sai ngữ cảnh."

    # Lưu lỗi vào lịch sử
    if user_id not in user_error_history:
        user_error_history[user_id] = []
    user_error_history[user_id].append({"type": "writing", "error": errors})

    # Xác định cấp độ
    user_errors = user_error_history[user_id]
    error_count = len(user_errors)
    level = "Người mới bắt đầu" if error_count > 5 else "Trung cấp" if error_count > 2 else "Nâng cao"

    # Gợi ý tài liệu
    suggestion = "Học khóa học ngữ pháp trên Coursera." if error_type == "Ngữ pháp" else "Sử dụng Quizlet để học từ vựng."
    if semantic_error:
        suggestion += " Xem khóa học viết nâng cao trên Coursera."
    if level == "Người mới bắt đầu":
        suggestion += " Bắt đầu với Duolingo."

    return jsonify({
        "error": True,
        "errors": errors,
        "error_type": error_type,
        "reason": reason,
        "corrected": corrected,
        "suggestion": suggestion,
        "level": level
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)