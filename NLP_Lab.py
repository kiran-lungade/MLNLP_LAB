# import spacy
# nlp=spacy.load("en_core_web_sm")
# from collections import Counter
# text=("abc. abc! is very a stop was is a been sport")

# about_doc=nlp(text)
# sentences=list(about_doc.sents)
# len(sentences)

# for sentence in sentences:
# 	print(f"{sentence[:5]}...")

# print("*******************************************")
# for token in about_doc:
# 	print(token,token.idx)

# print("*******************************************")
# spacy_stop=spacy.lang.en.stop_words.STOP_WORDS
# len(spacy_stop)
# for stop_word in list(spacy_stop)[:15]:
# 	print(stop_word)
# print([token for token in about_doc if not token.is_stop])

# print("*****************************************************")
# for token in about_doc:
# 	if str(token)!=str(token.lemma_):
# 		print(f"{str(token):>20}:{str(token.lemma_)}")

# print("************************************************************")
# words=[
# 	token.text
# 	for token in about_doc
# 	if not token.is_stop and not token.is_punct
# ]
# print(Counter(words).most_common(5))

# print("**************************************")
# for token in about_doc:
# 	print(f"""
# 	   TOKEN: {str(token)}
# ------------------
# 		TAG: {str(token.tag)}
# 		POS: {str(token.pos_)}
# 		Explin: {spacy.explain(token.tag_)}
# 	   """)











# import spacy
# from spacy import displacy

# nlp=spacy.load("en_core_web_sm")
# text=("a b c d e.")
# doc=nlp(text)

# for token in doc:
#     print(f"""
#           token: {token.text}
#           ********************
#           {token.tag_}
#           *******************
#           {token.head.text}
#           **********************
#           {token.dep_}""")
# displacy.serve(doc, style="dep")













# import nltk
# from nltk.util import ngrams

# sent='a d v r w e f s f a.'

# n=1
# u=ngrams(sent.split(),n)
# for i in u:
#     print(i)
# n=2
# u=ngrams(sent.split(),n)
# for i in u:
#     print(i)
# n=3
# u=ngrams(sent.split(),n)
# for i in u:
#     print(i)
















# import spacy
# nlp=spacy.load("en_core_web_sm")
# def p(text):
#       doc=nlp(text)
#       entities=[(ent.text, ent.label_) for ent in doc.ents]
#       return entities

# text=" Maharashtra Godavari. India Sun"

# ne=p(text)
# print("NAMES:: ")
# for entity, l in ne:
#     print(f"{entity}: {l}")














# import spacy
# import re

# # Load the spaCy English language model
# nlp = spacy.load("en_core_web_sm")

# # Define regular expressions
# url_pattern = re.compile(r'https?://\S+|www\.\S+')

# ip_address_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

# date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
# pan_number_pattern = re.compile(r'[A-Z]{5}[0-9]{4}[A-Z]')

# def extract_entities(text):
#     # Tokenize the text using spaCy
#     doc = nlp(text)

#     # Find entities using regular expressions
#     urls = re.findall(url_pattern, text)
#     ip_addresses = re.findall(ip_address_pattern, text)
#     dates = re.findall(date_pattern, text)
#     pan_numbers = re.findall(pan_number_pattern, text)

#     # Extract spaCy entities
#     entities = [(ent.text, ent.label_) for ent in doc.ents]

#     return {
#         'urls': urls,
#         'ip_addresses': ip_addresses,
#         'dates': dates,
#         'pan_numbers': pan_numbers,
#         'spaCy_entities': entities
#     }
# # Example usage
# text_data = """
# Here is a sample text with a URL: https://www.Sample.com.
# Also, an IP address: 192.168.789.102.
# The date is 2023-01-01.
# A PAN number is BBRPL4574H.
# """

# results = extract_entities(text_data)

# print("URLs:", results['urls'])
# print("IP Addresses:", results['ip_addresses'])
# print("Dates:", results['dates'])
# print("PAN Numbers:", results['pan_numbers'])
# print("Entities:", results['spaCy_entities'])


















# import numpy as np
# from gensim.utils import simple_preprocess
# from gensim import corpora
# from gensim import models

# text2 = open('kiranlungade.txt', encoding ='utf-8')

# tokens2 =[]
# for line in text2.read().split('.'):
#   tokens2.append(simple_preprocess(line, deacc = True))

# g_dict2 = corpora.Dictionary(tokens2)

# print("The dictionary has: " +str(len(g_dict2)) + " tokens\n")
# print(g_dict2.token2id)
# g_bow =[g_dict2.doc2bow(token, allow_update = True) for token in tokens2]
# print("\n Bag of Words : ", g_bow)
# print("\n")
# print("--------------------------------------------------------------------------------------")

# print("--------------------------------------TFIDF VECTOR------------------------------------")


# ##TFIDF
# text = [ "abc abc abc."]

# g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
# g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

# print("Dictionary : ")
# for item in g_bow:
#     print([[g_dict[id], freq] for id, freq in item])

# g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

# print("\n TF-IDF Vector:")
# for item in g_tfidf[g_bow]:
#     print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])
