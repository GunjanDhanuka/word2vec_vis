import plotly
import plotly.graph_objs as go
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import pandas as pd
from scripts.plots import horizontal_bar, display_scatterplot_2D, display_scatterplot_3D
from scripts.train import train_on_dataframe, train_on_raw_text
import requests

st.set_page_config(page_title="Word Embedding Visualizing Web App", page_icon="📊")

@st.cache
def download_models():
    with st.spinner("Downloading Pretrained Models. This might take a minute..."):
        download_file_from_google_drive("1a4PRwGYjvufC--5YaiZTZAvDWzLXgwtd", "imdb.bin")
        download_file_from_google_drive(
            "19AlUcJQDq8i_jMIZp_1cMISHIBMlAGCJ", "marvel.bin"
        )
        download_file_from_google_drive(
            "1jl1S3GFFKur0Gh5OASEoYtcR0B3hUr-0", "simpsons.bin"
        )


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


download_models()
MODEL_PATH = "imdb.bin"


def append_list(sim_words, words):

    list_of_words = []

    for i in range(len(sim_words)):

        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words


display_params = False
file_type = st.sidebar.selectbox(
    "Upload your own text corpus or use one of the pretrained ones.",
    ("Pretrained", "Paste your own text", "Upload text dataset"),
)
if file_type == "Pretrained":
    pretrained_data = st.sidebar.selectbox(
        "Choose desired corpus",
        ("IMDB Reviews", "Simpsons Dialogues", "Marvel Dialogues"),
    )
    if pretrained_data == "IMDB Reviews":
        MODEL_PATH = "imdb.bin"
        model = Word2Vec.load(MODEL_PATH)
        st.session_state["model"] = model
    elif pretrained_data == "Simpsons Dialogues":
        MODEL_PATH = "simpsons.bin"
        model = Word2Vec.load(MODEL_PATH)
        st.session_state["model"] = model
    elif pretrained_data == "Marvel Dialogues":
        MODEL_PATH = "marvel.bin"
        model = Word2Vec.load(MODEL_PATH)
        st.session_state["model"] = model
    display_params = True
    st.session_state["method"] = str(pretrained_data)
elif file_type == "Upload text dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file, CSV format for dataframes, or a single text file for raw text.",
        type=["csv", "txt"],
    )
    if uploaded_file is not None:
        file_details = {
            "FileName": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": uploaded_file.size,
        }
        # st.write(file_details)
        if file_details["FileType"] == "text/csv":
            print("Uploaded CSV File")
            dataframe = pd.read_csv(uploaded_file)
            text_col = st.sidebar.selectbox(
                "Choose the column which contains text", dataframe.columns
            )
            PARAMS = dict()
            with st.form(key="train_params_1"):
                min_count = st.slider(
                    "Ignores all words with total frequency lower than this", 1, 50, 20
                )
                window = st.slider(
                    "Maximum distance between the current and predicted word within a sentence.",
                    1,
                    10,
                    2,
                )
                sg_choice = st.selectbox("Training Algorithm", ("Skip-gram", "CBOW"))
                vector_size = st.slider(
                    "Dimensionality of the word vectors", 50, 300, 300
                )
                alpha = st.slider("Initial learning rate", 0.001, 0.1, 0.03)
                min_alpha = st.slider(
                    "Learning rate with linearly drop to min_alpha as training progresses",
                    0.00001,
                    0.001,
                    0.0007,
                )
                negative = st.slider(
                    "If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn",
                    5,
                    20,
                    20,
                )
                epochs = st.slider(
                    "Number of epochs for training (more the epochs, greater the training time)",
                    5,
                    30,
                    10,
                )
                submit_button = st.form_submit_button(label="Train Word2Vec Model")

            if submit_button:
                PARAMS["min_count"] = min_count
                PARAMS["window"] = window
                PARAMS["sg_choice"] = sg_choice
                PARAMS["vector_size"] = vector_size
                PARAMS["alpha"] = alpha
                PARAMS["min_alpha"] = min_alpha
                PARAMS["negative"] = negative
                PARAMS["epochs"] = epochs
                with st.spinner(
                    "Please wait while we prepare the data and train the model..."
                ):
                    model = train_on_dataframe(dataframe, text_col, PARAMS)
                    st.session_state["model"] = model
                    st.session_state["method"] = "dataframe"
                print("Exited model training...")
        elif file_details["FileType"] == "text/plain":
            print("Uploaded .TXT file")
            raw_text = str(uploaded_file.read(), "utf-8")
            PARAMS = dict()
            with st.form(key="train_params_2"):
                min_count = st.slider(
                    "Ignores all words with total frequency lower than this", 1, 50, 20
                )
                window = st.slider(
                    "Maximum distance between the current and predicted word within a sentence.",
                    1,
                    10,
                    2,
                )
                sg_choice = st.selectbox("Training Algorithm", ("Skip-gram", "CBOW"))
                vector_size = st.slider(
                    "Dimensionality of the word vectors", 50, 300, 300
                )
                alpha = st.slider("Initial learning rate", 0.001, 0.1, 0.03)
                min_alpha = st.slider(
                    "Learning rate with linearly drop to min_alpha as training progresses",
                    0.00001,
                    0.001,
                    0.0007,
                )
                negative = st.slider(
                    "If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn",
                    5,
                    20,
                    20,
                )
                epochs = st.slider(
                    "Number of epochs for training (more the epochs, greater the training time)",
                    5,
                    30,
                    10,
                )
                submit_button = st.form_submit_button(label="Train Word2Vec Model")

            if submit_button:
                PARAMS["min_count"] = min_count
                PARAMS["window"] = window
                PARAMS["sg_choice"] = sg_choice
                PARAMS["vector_size"] = vector_size
                PARAMS["alpha"] = alpha
                PARAMS["min_alpha"] = min_alpha
                PARAMS["negative"] = negative
                PARAMS["epochs"] = epochs
                with st.spinner(
                    "Please wait while we prepare the data and train the model..."
                ):
                    model = train_on_raw_text(raw_text, PARAMS)
                    st.session_state["model"] = model
                    st.session_state["method"] = "text_file"
                print("Exited model training...")
    display_params = True
elif file_type == "Paste your own text":
    print("Pasting your text option chosen")
    raw_text = st.text_area("Paste your own text here")
    PARAMS = dict()
    with st.form(key="train_params_3"):
        min_count = st.slider(
            "Ignores all words with total frequency lower than this", 1, 10, 2
        )
        window = st.slider(
            "Maximum distance between the current and predicted word within a sentence.",
            10,
            100,
            50,
        )
        sg_choice = st.selectbox("Training Algorithm", ("Skip-gram", "CBOW"))
        vector_size = st.slider("Dimensionality of the word vectors", 50, 300, 300)
        alpha = st.slider("Initial learning rate", 0.001, 0.1, 0.03)
        min_alpha = st.slider(
            "Learning rate with linearly drop to min_alpha as training progresses",
            0.00001,
            0.001,
            0.0007,
        )
        negative = st.slider(
            "If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn",
            5,
            20,
            20,
        )
        epochs = st.slider(
            "Number of epochs for training (more the epochs, greater the training time)",
            5,
            30,
            30,
        )
        submit_button = st.form_submit_button(label="Train Word2Vec Model")

    if submit_button:
        PARAMS["min_count"] = min_count
        PARAMS["window"] = window
        PARAMS["sg_choice"] = sg_choice
        PARAMS["vector_size"] = vector_size
        PARAMS["alpha"] = alpha
        PARAMS["min_alpha"] = min_alpha
        PARAMS["negative"] = negative
        PARAMS["epochs"] = epochs
        with st.spinner("Please wait while we prepare the data and train the model..."):
            model = train_on_raw_text(raw_text, PARAMS)
            st.session_state["model"] = model
            st.session_state["method"] = "paste"
        print("Exited model training...")
    display_params = True


if "model" in st.session_state:
    print("using saved model")
    print("Method of ", st.session_state["method"])
    model = st.session_state["model"]

if display_params:
    print("Entered display params...")
    dim_red = st.sidebar.selectbox("Select dimension reduction method", ("PCA", "TSNE"))
    dimension = st.sidebar.selectbox(
        "Select the dimension of the visualization", ("2D", "3D")
    )
    user_input = st.sidebar.text_input(
        "Type the word that you want to investigate. You can type more than one word by separating one word with other with comma (,)",
        "",
    )
    top_n = st.sidebar.slider(
        "Select the amount of words associated with the input words you want to visualize ",
        5,
        100,
        (5),
    )
    annotation = st.sidebar.radio(
        "Enable or disable the annotation on the visualization", ("On", "Off")
    )

    if dim_red == "TSNE":
        perplexity = st.sidebar.slider(
            "Adjust the perplexity. The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity",
            5,
            50,
            (30),
        )

        learning_rate = st.sidebar.slider("Adjust the learning rate", 10, 1000, (200))

        iteration = st.sidebar.slider(
            "Adjust the number of iteration", 250, 100000, (1000)
        )

    else:
        perplexity = 0
        learning_rate = 0
        iteration = 0

    if user_input == "":
        similar_word = None
        labels = None
        color_map = None

    else:

        user_input = [x.strip() for x in user_input.split(",")]
        result_word = []

        for words in user_input:
            if words not in model.wv.key_to_index:
                st.sidebar.warning(f"{words} is not in the vocabulary.")
                user_input.remove(words)
                continue
            sim_words = model.wv.most_similar(words, topn=top_n)
            sim_words = append_list(sim_words, words)

            result_word.extend(sim_words)

        similar_word = [word[0] for word in result_word]
        similarity = [word[1] for word in result_word]
        similar_word.extend(user_input)
        labels = [word[2] for word in result_word]
        label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
        color_map = [label_dict[x] for x in labels]

st.title("Word Embedding Visualization using Word2Vec Model")

st.header("How to use the web app?")
st.write(
    """On the sidebar,\n
    1. First choose the data you want to use for visualization.
    2. Then, if you upload a custom dataset or paste your own text.
        you can choose the parameters for training the model.
    3. Then, you can choose the dimensionality reduction technique - either PCA or TSNE.
    4. Then, you can choose the dimension of the visualization - either 2D or 3D.
    5. Then, you can type the word(s) that you want to investigate.
    6. Then, you can select the amount of words associated with the input words you want to visualize.

    Lastly, you have an option to enable or disable the text annotations.
    """
)

if dimension == "2D":
    st.header("2D Visualization")
    st.write(
        "For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization."
    )
    display_scatterplot_2D(
        model,
        user_input,
        similar_word,
        labels,
        color_map,
        annotation,
        dim_red,
        perplexity,
        learning_rate,
        iteration,
        top_n,
    )
else:
    st.header("3D Visualization")
    st.write(
        "For more detail about each point (just in case it is difficult to read the annotation), you can hover around each points to see the words. You can expand the visualization by clicking expand symbol in the top right corner of the visualization."
    )
    display_scatterplot_3D(
        model,
        user_input,
        similar_word,
        labels,
        color_map,
        annotation,
        dim_red,
        perplexity,
        learning_rate,
        iteration,
        top_n,
    )

st.header("The Top 5 Most Similar Words for Each Input")
count = 0
for i in range(len(user_input)):

    st.write("The most similar words from " + str(user_input[i]) + " are:")
    horizontal_bar(similar_word[count : count + 5], similarity[count : count + 5])

    count = count + top_n
