import pandas as pd
import string
import re
import joblib
import matplotlib.pyplot as plt
import underthesea
from underthesea import text_normalize

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
import gradio as gr

def handle_repeated_syllables(text):
    # Sử dụng regex để tìm các từ có âm tiết lặp lại (ví dụ: quááááá)
    repeated_syllables_pattern = re.compile(r'(\w+?)\1+', re.UNICODE)
    # Hàm xử lý việc loại bỏ âm tiết lặp lại
    def handle_repetition(match):
        word = match.group(1)
        # Giữ lại chỉ một phần lặp lại và thêm vào từ gốc
        return word

    # Áp dụng hàm xử lý vào chuỗi
    processed_text = repeated_syllables_pattern.sub(handle_repetition, text)
    return processed_text

EMAIL = re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})")
URL = re.compile(r"https?:\/\/(?!.*:\/\/)\S+")
PHONE = re.compile(r"(09|01[2|6|8|9])+([0-9]{9})\b")
MENTION = re.compile(r"@.+?:")
NUMBER = re.compile(r'\b\d+\S*\b')
DATETIME = '\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}'

# Delete price, 3g/4g/5g
PRICE = r'\b\d{1,4}(?:\.\d{3})*(?:\.\d+)?(?:[ktrđg])\b'

def replace_common_token(txt):
    txt = re.sub(EMAIL, ' ', txt)
    txt = re.sub(URL, ' ', txt)
    txt = re.sub(MENTION, ' ', txt)
    txt = re.sub(DATETIME, ' ', txt)
    txt = re.sub(NUMBER, ' ', txt)
    txt = re.sub(PRICE, ' ', txt)
    return txt

def remove_unnecessary_characters(text):
    RE_CLEAR = re.compile("[\n\r]+")# Thay thế các chuỗi xuống dòng (\n hoặc \r) bằng một ký tự trắng
    text = re.sub(RE_CLEAR, ' ', text)
    # Sử dụng string.punctuation để lấy tất cả các ký tự dấu câu
    translator = str.maketrans('', '', string.punctuation)
    # Loại bỏ dấu câu từ văn bản sử dụng bảng dịch (translator)
    text = text.translate(translator)

    return text

def normalize_acronyms(text, teencode_file='teencode.xlsx'):
    # Đọc dữ liệu từ tệp Excel teencode.xlsx
    teencode_df = pd.read_excel(teencode_file, header=None, names=['teencode', 'replace'])

    words = []
    for word in text.strip().split():
        word = word.strip(string.punctuation)
        # Tìm kiếm trong teencode_df và thay thế
        replacement = teencode_df.loc[teencode_df['teencode'].str.lower() == word, 'replace'].values
        if len(replacement) > 0:
            words.append(replacement[0])
        else:
            words.append(word)

    return ' '.join(words)

stopword = []
with open('stopword_train.txt', 'r', encoding='utf8') as fp:
    for line in fp.readlines():
        stopword.append(line.strip())
len(stopword)

# loại stopword khỏi dữ liệu
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)

# Không tiến hành tách từ dính nhau vì bộ dữ liệu từ dính nhau chỉ được áp dụng cho bộ dữ liệu cũ, không bao quát cho dữ liệu mới nhập vào.
def text_preprocess(text):
    # 1. Chuẩn hóa văn bản tiếng việt
    text = text_normalize(text)
    # 2. Xử lý láy âm tiết
    text = handle_repeated_syllables(text)
    # 3. Loại bỏ các common token
    text = replace_common_token(text)
    # 4. Xóa bỏ dấu câu
    text = remove_unnecessary_characters(text)
    # 5. Đưa về lower
    text = text.lower()
    # 6. Chuẩn hóa các từ viết tắt cơ bản
    text = normalize_acronyms(text)
    # 7. Loại bỏ các stopword tiếng Việt
    text = remove_stopwords(text)
    # 8. Loại bỏ các khoảng trắng liên tiếp
    RE_CLEAR = re.compile("\s+") # Các khoảng trắng liên tiếp
    text = re.sub(RE_CLEAR,' ', text)
    return text

def predict_rate(text):
    text = text_preprocess(text)
    text = underthesea.word_tokenize(text, format="text")

    balanced_df = pd.read_csv('balanced_df.csv')
    X, y = balanced_df['comment_token'], balanced_df['n_star']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    count_vectorizer = CountVectorizer(ngram_range=(1, 5), stop_words=stopword, max_df=0.5, min_df=5)
    tfidf_vectorizer = TfidfTransformer(use_idf=False, sublinear_tf = True, norm='l2', smooth_idf=True)

    X_train = count_vectorizer.fit_transform(X_train)
    X_train = tfidf_vectorizer.fit_transform(X_train)

    X_test = count_vectorizer.transform([text])
    X_test = tfidf_vectorizer.transform(X_test)
    # Dự đoán dữ liệu mới
    maxent_model = joblib.load( 'maxent_model.pkl')
    y_pre = maxent_model.predict(X_test)
    return str(y_pre[0])


def combine(a):
    return predict_rate(a) + '⭐' # Chỗ này điền dự đoán của mô hình

with gr.Blocks() as demo:

    txt = gr.Textbox(label="Bạn thấy sản phẩm điện thoại này như thế nào", lines=2)
    txt_2 = gr.Textbox(value="", label="Kết quả đánh giá")
    btn = gr.Button(value="Submit")
    btn.click(combine, inputs=[txt], outputs=[txt_2])
    gr.Markdown("## Đánh giá ví dụ")
    gr.Examples(
        ["Tôi thấy sản phẩm này đẹp", "Cấu hình máy mạnh", "Máy chạy quá chậm"],
        [txt],
        txt_2,
        combine,
        cache_examples=True)

demo.launch(share=True)

