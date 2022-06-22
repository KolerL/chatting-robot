from PyQt5 import QtWidgets
import robot
import csv
from fuzzywuzzy import fuzz
from sklearn.naive_bayes import MultinomialNB
import pickle #模型储存取出
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
from selenium import webdriver

#字符串匹配
greeting_q = []
greeting_a = []
with open("greeting.csv", 'r',encoding='GBK') as f:
    greeting = csv.reader(f)
    header = next(greeting)
    for words in greeting:
        greeting_q.append(words[0])
        greeting_a.append(words[1])

#模糊匹配
def get_greeting(input_questions,question,answer):
    text = {}
    for key, value in enumerate(question):
        similarity = fuzz.partial_ratio(input_questions, value)
        if similarity > 33:
            text[key] = similarity
    if len(text) > 0:
        train = sorted(text.items(), key=lambda d: d[1], reverse=True)
        answer3 = answer[train[0][0]]
    else:
        answer3 = None
    return  answer3

# 停用词
stpwrdpath = "D:\PycharmFiles\chatting_robot_fin\stop_words.txt" #此处需要修改为保存的地址
stpwrd_dic = open(stpwrdpath, 'r',encoding = 'utf-8')
stpwrd_content = stpwrd_dic.read()

#结巴分词字符串处理：（空格分隔，字符串转换，去除停用词）
def jieba_text(text):
    text_jieba = jieba.cut(text)
    text_str = ""
    for word in text_jieba:
        text_str +=word+" "
    text_list = text_str.split('\n')#空格分隔，换行符添加
    text_apply = []
    for file in text_list:
        for word in file:
            if word in stpwrd_content:
                file = file.replace(word,'')#停用词替换
            else:
                continue
        text_apply.append(file)
    return [text_list,text_apply] #text_list：分隔后文本 text_apply:分隔处理，去除停用词操作

#tf-idf数学处理
def tf_idf(text_train,text_test): #text_train:训练样本处理 text_test: 预测样本处理
    vectorizer = CountVectorizer(min_df=1,max_features= 6)
    transformer = TfidfTransformer()
    tfidf_train = transformer.fit_transform(vectorizer.fit_transform(text_train))
    tfidf_test = transformer.fit_transform(vectorizer.transform(text_test))

    train_array = tfidf_train.toarray()
    test_array = tfidf_test.toarray()

    return [train_array,test_array] #train_array:处理后训练样本 test_array:处理后预测样本

#功能模块
question_dataset = []
answer_dataset = []
with open("dataset.csv", 'r', encoding='GBK') as f:
    dataset = csv.reader(f)
    header = next(dataset)
    for words in dataset:
        question_dataset.append(words[0])
        answer_dataset.append(words[1])

#打开网页
def web_open(result):
    global driver
    driver = webdriver.Chrome('D:\PycharmFiles\chatting_robot_fin\chromedriver.exe') #此处需要修改保存的地址
    driver.get(result)
    print(driver.page_source)

#贝叶斯模型训练
def bayes_model(dataset,label):
    model = MultinomialNB()
    model.fit(dataset, label)
    return model

#训练文件读取内容
train_data = ""
train_label = ""
with open("train_data.csv", 'r',encoding='GBK') as f:
    train= csv.reader(f)
    header = next(train)
    for words in train:
        train_data += words[0]+'\n'
        train_label += words[1]+'\n'
train_data = train_data.strip('\n')
train_label = train_label.strip('\n')

#贝叶斯分类器训练
train_data_apply = jieba_text(train_data)[1]
train_label_apply = jieba_text(train_label)[0]
train_array = tf_idf(train_data_apply,['你好'])[0]
model_apply = bayes_model(train_array,train_label_apply)

#清除操作
def off_click(self):
    ui.textEdit.clear()#清除聊天框

#发送操作
def on_click(self):
    text_1 = ui.textEdit.toPlainText() #输入
    ui.textBrowser_2.setText(text_1)
    input_list = jieba_text(text_1)[1]
    input_apply = tf_idf(train_data_apply, input_list)[1]
    value = model_apply.predict(input_apply)

    if value == [' A ']:
        reply = get_greeting(text_1, greeting_q, greeting_a)
        if reply == None:
            reply_none = '不好意思，没有明白您的意思嗷'
            ui.textBrowser.setText(reply_none)
        else:
            ui.textBrowser.setText(str(reply))
    if value == [' B ']:
        reply = get_greeting(text_1, question_dataset, answer_dataset)
        if reply == None:
            reply_none = '不好意思，没有明白您的意思嗷'
            ui.textBrowser.setText(reply_none)
        else:
            ui.textBrowser.setText('请查收有关信息(:-D)')
            web_open(str(reply))
    if value == [' D ']:
        reply = get_greeting(text_1, question_dataset, answer_dataset)
        if reply == None:
            reply_none = '不好意思，没有明白您的意思嗷'
            ui.textBrowser.setText(reply_none)
        else:
            web_open(str(reply))
            ui.textBrowser.setText('请查收有关信息(:-D)')

app = QtWidgets.QApplication([])
window = QtWidgets.QMainWindow()
ui = robot.Ui_MainWindow()
ui.setupUi(window)  # 启动运行
ui.pushButton.clicked.connect(on_click)
ui.pushButton_2.clicked.connect(off_click)
window.show()  # 显示窗口
app.exec()