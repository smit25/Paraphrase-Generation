import spacy

nlp = spacy.load('en_core_web_sm')

for _ in range():
    doc = nlp()
    s = list(doc)
    tmp,temp,sub = "","",-1
    for i in doc:
        if i.pos_ == 'VERB':
            s[i.i] = i
        elif i.dep_ == 'nsubj':
            sub = i.i
            temp = i
        elif i.dep_ == 'dobj':
            tmp = i.text.capitalize()
            s[i.i] = temp
            s.insert(i.i,"by")

s[sub] = tmp
print(' '.join(str(e) for e in s))