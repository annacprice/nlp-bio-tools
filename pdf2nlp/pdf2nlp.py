#!/usr/bin/env python2
#----------------------------------------------------------------------
# Author: Anna Price

# This script is part of the nlp-bio-tools software tool: 
# https://github.com/annacprice/nlp-bio-tools

# It takes the input pdf files and converts them to plain text,
# the text is then passed through a natural language processing
# pipeline and the final output for each pdf is saved as
# individual .txt files

# This script is designed to be used with the accompanying Dockerfile.
# The input pdf articles should be placed in data/papers.
# The output txt files are saved to data/output local machine.

# To build the Docker image from current directory:
# docker build -t pdf2nlp .

# To run the program in the Docker container from current directory:
# docker run -v $(pwd)/data:/data --rm pdf2nlp
#----------------------------------------------------------------------
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
from os import listdir
from os.path import isfile, join
import sys, getopt
import nltk, re
from nltk import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer


# Define stemmers
snowball = SnowballStemmer("english")
porter = PorterStemmer()

def convert_pdf(fname, pages=None):
    # mines pdf, returns the text as a string
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = "utf-8"

    # layout analysis
    laparams = LAParams(all_texts=True, detect_vertical=False,
                        line_overlap=0.5, char_margin=2.0, line_margin=0.5,\
                        word_margin=0.1, boxes_flow=0.5)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    infile = file(fname, "rb")

    for page in PDFPage.get_pages(infile, pagenums, check_extractable=False):
        interpreter.process_page(page)
    infile.close()
    device.close()
    message = retstr.getvalue()
    retstr.close
    return message

def replace_all(text, dict):
    # define dictionary
    for gb, us in dict.items():
        text = text.replace(us, gb)
    return text

def txt_process(in_pdf, out_txt):
    # NLP pipeline
    for pdf in os.listdir(in_pdf):
        fileExtension = pdf.split(".")[-1]
        if fileExtension == "pdf":
            pdfFile = in_pdf + pdf
            try:
                message = convert_pdf(pdfFile)
            except:
                continue
        
            # STEP 1: TOKENISE
            message = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", message)
            message = " ".join(str(e) for e in message)
            message = message.lower()
            
            # STEP 2: CONVERT US ENGLISH TO BRITISH ENGLISH
            us2gb = {}
            # Replace US English with UK English
            with open("/data/ustouk.txt", "r") as file:
                for line in file:
                    (key, val) = line.split()
                    us2gb[key] = val
            message = replace_all(message, us2gb)
            
            # STEP 3: REMOVE PUNCTUATION & NUMBERS
            # remove punctuation
            message = re.sub(r"[^\w\s]+", "", message)
            # remove underscore
            message = re.sub(r"\_", "", message)
            # remove combinations of letters+numbers
            message = re.sub(r"\w*[\d.\-]\w*", "", message)
            # remove standalone numbers
            message = re.sub(r"\b\d+\b", "", message)
            
            # STEP 4: REMOVE STOP WORDS
            message = nltk.word_tokenize(message)

            # remove common stopwords
            stopwords = nltk.corpus.stopwords.words()
            filteredWords = []
            for word in message:
                if word not in stopwords:
                    filteredWords.append(word)
                    message = filteredWords
            
            # remove biomedical academic stopwords
            with open("/data/biostopwords.txt") as file:
                stopbio = file.read().splitlines()

            message = [item for item in message if item not in stopbio]

            # STEP 5: STEMMING
            stemmedWords = []
            for word in message:
                stemmedWords.append(snowball.stem(word))
                message = stemmedWords
            
            message = [x.encode("utf-8") for x in message]
            message = " ".join(str(e) for e in message)
            message = " ".join([w for w in message.split() \
                                if len(w)>2])
                                
            # Save results to data/output on local machine
            txtFile = out_txt + pdf + ".txt"
            txtFile = open(txtFile, "w")
            txtFile.write(message)

if __name__ == "__main__":
    # Path for the input pdf files and output txt files
    in_pdf = "/data/papers/"
    out_txt = "/data/output/"
    
    txt_process(in_pdf, out_txt)
