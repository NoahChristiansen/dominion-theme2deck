import pandas as pd
import numpy as np
import streamlit as st

import gensim.downloader as api
from gensim.models import KeyedVectors

## run with: streamlit run theme2deck.py

st.title("Dominion Themed Deck Finder")


cards = pd.read_csv('./data/cleaned_cards.csv')
cards['split_name'] = cards['clean_name'].replace({'moneylender':'money lender banker',
                                                   'cutpurse':'cut purse thief',
                                                   'salvager':'salvage recover',
                                                   'haggler':'haggle barter',
                                                   'graverobber':'grave robber',
                                                   'feodum':'feudal estate',
                                                   'transmogrify':'transform'}).str.split()


@st.cache(allow_output_mutation=True)
def load_model(name = "theme2deck.wordvectors", from_api = False):
    with st.spinner('Downloading word2vec model... please hold...'):
        if from_api:
            model = api.load(name)
        else:
            model = KeyedVectors.load(name)
    return model

def get_card_sim(query, model):
    query_df = cards.copy()
    query_df['word_sim'] = [[(word,model.similarity(query, word))
                             for word in card
                             if word in model
                            ] for card in query_df['split_name']]
    query_df['similarity'] = [np.mean([np.max([word[1] for word in card])]*5
                                      +[np.mean([word[1] for word in card])])
                              if len(card)>=1 else -1
                              for card in query_df['word_sim']]

    return query_df.sort_values('similarity', ascending = False).reset_index(drop = True)

def main():
    model_name = st.selectbox("Which model would you like to use?",
                              ['theme2deck.wordvectors']+list(api.info()['models'].keys()))
    st.write(f"Model used: {model_name}")
    if model_name in list(api.info()['models'].keys()):
        model = load_model(name = model_name, from_api = True)
    else:
        model = load_model(name = model_name, from_api = False)
        
    st.success("Model loaded!")
    
    user_input = st.text_input("Theme Idea:")
    num_cards  = st.number_input("How many cards?",min_value = 5, value = 20)

    if user_input:
        model_response = get_card_sim(user_input, model).head(num_cards)
        
        images = [list(model_response['images'].values)[i:i+5] for i in range(0, len(model_response['images']), 5)]
        names  = [list(model_response['card_name'])[i:i+5] for i in range(0, len(model_response['card_name']), 5)]
        
        for i in range(len(images)):
            st.image(images[i], width = 137, caption = names[i])
        
        st.subheader("Most Similar Cards:")
        st.write(model_response[['card_name','set_name','type','cost','card_text','similarity']].head(num_cards))
        
    st.write("Card data from: https://github.com/wesbuck/DominionCardAPI")
    st.write("Card images from: https://github.com/tempfillernamegithq/dominion-cards")
    st.write("with additional images scraped from: http://wiki.dominionstrategy.com/index.php/Category:Card_images")

main()