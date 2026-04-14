import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")
st.title("Next Word Predictor")
st.write("Trained on CampusX DSMP FAQ data.")

faqs = """About the Program
What is the course fee for  Data Science Mentorship Program (DSMP 2023)
The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
What is the total duration of the course?
The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
What is the syllabus of the mentorship program?
We will be covering the following modules:
Python Fundamentals
Python libraries for Data Science
Data Analysis
SQL for Data Science
Maths for Machine Learning
ML Algorithms
Practical ML
MLOPs
Case studies
You can check the detailed syllabus here - https://learnwith.campusx.in/courses/CampusX-Data-Science-Mentorship-Program-637339afe4b0615a1bbed390
Will Deep Learning and NLP be a part of this program?
No, NLP and Deep Learning both are not a part of this program's curriculum.
What if I miss a live session? Will I get a recording of the session?
Yes all our sessions are recorded, so even if you miss a session you can go back and watch the recording.
Where can I find the class schedule?
Checkout this google sheet to see month by month time table of the course - https://docs.google.com/spreadsheets/d/16OoTax_A6ORAeCg4emgexhqqPv3noQPYKU7RJ6ArOzk/edit?usp=sharing.
What is the time duration of all the live sessions?
Roughly, all the sessions last 2 hours.
What is the language spoken by the instructor during the sessions?
Hinglish
How will I be informed about the upcoming class?
You will get a mail from our side before every paid session once you become a paid user.
Can I do this course if I am from a non-tech background?
Yes, absolutely.
I am late, can I join the program in the middle?
Absolutely, you can join the program anytime.
If I join/pay in the middle, will I be able to see all the past lectures?
Yes, once you make the payment you will be able to see all the past content in your dashboard.
Where do I have to submit the task?
You don't have to submit the task. We will provide you with the solutions, you have to self evaluate the task yourself.
Will we do case studies in the program?
Yes.
Where can we contact you?
You can mail us at nitish.campusx@gmail.com
Payment/Registration related questions
Where do we have to make our payments? Your YouTube channel or website?
You have to make all your monthly payments on our website. Here is the link for our website - https://learnwith.campusx.in/
Can we pay the entire amount of Rs 5600 all at once?
Unfortunately no, the program follows a monthly subscription model.
What is the validity of monthly subscription? Suppose if I pay on 15th Jan, then do I have to pay again on 1st Feb or 15th Feb
15th Feb. The validity period is 30 days from the day you make the payment. So essentially you can join anytime you don't have to wait for a month to end.
What if I don't like the course after making the payment. What is the refund policy?
You get a 7 days refund period from the day you have made the payment.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmail.com
Post registration queries
Till when can I view the paid videos on the website?
This one is tricky, so read carefully. You can watch the videos till your subscription is valid. Suppose you have purchased subscription on 21st Jan, you will be able to watch all the past paid sessions in the period of 21st Jan to 20th Feb. But after 21st Feb you will have to purchase the subscription again.
But once the course is over and you have paid us Rs 5600(or 7 installments of Rs 799) you will be able to watch the paid sessions till Aug 2024.
Why lifetime validity is not provided?
Because of the low course fee.
Where can I reach out in case of a doubt after the session?
You will have to fill a google form provided in your dashboard and our team will contact you for a 1 on 1 doubt clearance session
If I join the program late, can I still ask past week doubts?
Yes, just select past week doubt in the doubt clearance google form.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmai.com
Certificate and Placement Assistance related queries
What is the criteria to get the certificate?
There are 2 criterias:
You have to pay the entire fee of Rs 5600
You have to attempt all the course assessments.
I am joining late. How can I pay payment of the earlier months?
You will get a link to pay fee of earlier months in your dashboard once you pay for the current month.
I have read that Placement assistance is a part of this program. What comes under Placement assistance?
This is to clarify that Placement assistance does not mean Placement guarantee. So we dont guarantee you any jobs or for that matter even interview calls. So if you are planning to join this course just for placements, I am afraid you will be disappointed. Here is what comes under placement assistance
Portfolio Building sessions
Soft skill sessions
Sessions with industry mentors
Discussion on Job hunting strategies
"""

@st.cache_resource
def train_model():
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([faqs])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for sentence in faqs.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i+1])

    max_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_len, padding='pre'))

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_len - 1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)

    return model, tokenizer, max_len

def predict_next_words(seed_text, model, tokenizer, max_len, n):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if not token_list:
        return ["word not found in training data"]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-n:][::-1]
    results = []
    for idx in top_indices:
        for word, i in tokenizer.word_index.items():
            if i == idx:
                results.append(word)
                break
    return results

st.markdown("---")

if "ready" not in st.session_state:
    st.session_state.ready = False

num_words = st.sidebar.slider("Number of suggestions", 1, 5, 3)

if st.button("Train Model"):
    with st.spinner("Training... please wait"):
        m, t, ml = train_model()
        st.session_state.m     = m
        st.session_state.t     = t
        st.session_state.ml    = ml
        st.session_state.ready = True
    st.success("Model trained successfully!")

st.markdown("---")

if st.session_state.ready:
    user_input = st.text_input("Enter your text:")
    if user_input and user_input.strip():
        suggestions = predict_next_words(
            user_input.strip(),
            st.session_state.m,
            st.session_state.t,
            st.session_state.ml,
            num_words
        )
        st.write("**Predicted next words:**")
        for i, word in enumerate(suggestions):
            st.write(f"{i+1}. {word}")
else:
    st.info("Click Train Model to get started.")