import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


text = """ From there, programmers choose a machine learning model to use, supply the data, and let the computer model train itself to find patterns or make predictions. Over time the human programmer can also tweak the model, including changing its parameters, to help push it toward more accurate results. (Research scientist Janelle Shane’s website AI Weirdness is an entertaining look at how machine learning algorithms learn and how they can get things wrong — as happened when an algorithm tried to generate recipes and created Chocolate Chicken Chicken Cake.)

Some data is held out from the training data to be used as evaluation data, which tests how accurate the machine learning model is when it is shown new data. The result is a model that can be used in the future with different sets of data.

Successful machine learning algorithms can do different things, Malone wrote in a recent research brief about AI and the future of work that was co-authored by MIT professor and CSAIL director Daniela Rus and Robert Laubacher, the associate director of the MIT Center for Collective Intelligence."""
def summarizer(rawdocs):
    stopword = list(STOP_WORDS)
    # print(stopword)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)
    # print(doc)
    tokens= [token.text for token in doc]
    # print(tokens)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopword and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text]=1
            else:
                word_freq[word.text] +=1
    # print(word_freq) 


    max_freq = max(word_freq.values())
    # print(max_freq)

    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
        
    # print(word_freq)

    sent_tokens = [sent for sent in doc.sents]
    # print(sent_tokens)

    sent_scores= {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]                 
                else:
                    sent_scores[sent] += word_freq[word.text]
                
    # print(sent_scores)

    select_len = int(len(sent_tokens) * 0.3)
    # print(select_len)
    summary = nlargest(select_len,sent_scores, key = sent_scores.get)
    # print(summary)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    # print(text)
    # print(summary)
    # print(len(text.split(' ')))
    # print(len(summary.split(' ')))

    return summary, doc, len(rawdocs.split(' ')),len(summary.split(' '))

    