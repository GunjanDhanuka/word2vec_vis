![word2vec_vis](https://socialify.git.ci/GunjanDhanuka/word2vec_vis/image?description=1&descriptionEditable=Semantic%20word%20embeddings%20visualizer%20for%20your%20own%20text%20data.&font=Rokkitt&language=1&name=1&owner=1&pattern=Signal&theme=Light)

Check out the app live at Streamlit Cloud: [StreamLit link](https://gunjandhanuka-word2vec-vis-app-h90z23.streamlitapp.com/)

The webapp is based on the **Efficient Estimation of Word Representations in Vector Space** paper. Read it [here.](https://arxiv.org/abs/1301.3781)

### Screencast
[screencast.webm](https://user-images.githubusercontent.com/68523530/179350658-015f1fac-f5e8-4075-9e3a-12e09486904c.webm)

## Features:
1. Upload your own text corpus, or even a CSV dataset.
2. Train the Word2Vec on the fly using custom parameters.
3. Choose either PCA or TSNE as your dimensionality reduction technique.
4. Visualize the word in either 2-D or 3-D space.
5. Get similar words for each word, with similarity scores.
6. Option to tune the number of words you wish to see for each input.

## Steps to install locally:
1. Setup a virtual environment using Conda or any other method you prefer.
2. Install the dependencies from `requirements.txt`.
3. Run the following in the terminal.
    ```
    pip install -U pip setuptools wheel
    pip install -U spacy
    python -m spacy download en_core_web_sm
    ```
4. Run `streamlit run app.py` in the terminal to launch the web app.


