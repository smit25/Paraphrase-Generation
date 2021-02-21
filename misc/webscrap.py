from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pickle

final_list = set()
# Specify url of the web page
source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/10/1-10000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/10001-20000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/20001-30000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/30001-40000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/40001-50000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/50001-60000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/60001-70000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/70001-80000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/80001-90000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set

source = urlopen('https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2005/08/90001-100000').read()

# Make a soup 
soup = BeautifulSoup(source,'lxml')
soup

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
text = [val for pair in zip(paras, heads) for val in pair]
text = ' '.join(text)

# Drop footnote superscripts in brackets
text = re.sub(r"\[.*?\]+", '', text)
text = re.sub("[0123456789]", '', text)

# Replace '\n' (a new line) with '' and end the string at $1000.
text = text.replace('\n', '')[:-11]
text = text.split()
text = [i for i in text if i.isalnum()]
word_set = set()
for i in text:
  if i.isalpha:
    word_set.add(i)
final_list = final_list | word_set
print(len(final_list))

final_list = list(final_list)

with open('new_words.txt', 'wb') as new_save:
  pickle.dump(final_list, new_save)
  new_save.close



