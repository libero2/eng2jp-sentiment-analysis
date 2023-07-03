import codecs
from pdfminer.high_level import extract_text
from konoha import SentenceTokenizer
from transformers import pipeline
import stanza
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('in_JP', metavar='in_JP', type=str, help='Japanese document\'s path')
parser.add_argument('in_EN', metavar='in_EN', type=str, help='English document\'s path')
args = parser.parse_args()

in_JP = args.in_JP
in_EN = args.in_EN

def eng_jp_sentiment_analysis(in_JP, in_EN):

    out_JP = os.path.splitext(in_JP)[0] + '.txt'
    out_EN = os.path.splitext(in_EN)[0] + '.txt'
    tokenized_JP = os.path.splitext(in_JP)[0] + '_tokenized.txt'
    tokenized_EN = os.path.splitext(in_EN)[0] + '_tokenized.txt'

    def preprocessing(in_JP, in_EN):

        text_JP = extract_text(in_JP)
        text_EN = extract_text(in_EN)

        with open(str(out_JP), 'w+', encoding='utf-8') as textOut_JP:
            textOut_JP.write(text_JP)
        with open(str(out_EN), 'w+', encoding='utf-8') as textOut_EN:
            textOut_EN.write(text_EN)

    preprocessing(in_JP, in_EN)

    def sentence_segmentation(out_JP, out_EN):

        def jp_sentence_segmenter(out_JP, segmented_JP):

            text_in = codecs.open(out_JP, 'r', encoding='utf-8').read()

            tokenizer = SentenceTokenizer()
            tokenized = tokenizer.tokenize(text_in)

            with open(segmented_JP, 'w+', encoding='utf-8') as text_out:
                text_out.writelines(str(i) + '\n' for i in tokenized)
        
        def en_sentence_segmenter(out_EN, segmented_EN):

            text_in = codecs.open(out_EN, 'r', encoding='utf-8').read()
            segmenter = stanza.Pipeline('en')
            doc = segmenter(text_in)
            sentences = ([sentence.text for sentence in doc.sentences])
            with open(segmented_EN, 'w+', encoding='utf-8') as text_out:
                text_out.writelines(str(i) + '\n' for i in sentences)

        jp_sentence_segmenter(str(out_JP), str(tokenized_JP))
        en_sentence_segmenter(str(out_EN), str(tokenized_EN))

    sentence_segmentation(str(out_JP), str(out_EN))

    def sentence_sentiment_analyzer(tokenized_JP, tokenized_EN):

        sent_vals_JP = []
        sent_vals_EN = []

        dssc = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None
            )

        with open(tokenized_JP, 'r', encoding='utf-8') as lines_jp:
            for line in lines_jp:
                sent_vals_JP.append(dssc(line))
        with open(tokenized_EN, 'r', encoding='utf-8') as lines_en:
            for line in lines_en:
                sent_vals_EN.append(dssc(line))
        #add data funcs
        flatlist = [element for sublist in sent_vals_EN for element in sublist]
        flattest = [element for sublist in flatlist for element in sublist]

        df = pd.DataFrame.from_records(flattest)
        df1 = df.pivot(columns='label', values='score')

        i = 0
        s = 0
        vals = []
        while i <= len(df1.index) - 1:
            vals.append(s)
            i = i + 1
            if i % 3 == 0:
                s = s + 1
        print(vals)

        df1['index'] = vals

        collapsed_df = df1.groupby('index').first().reset_index()

        collapsed_df.to_csv('/Users/alef/Documents/codeStuff/jpyter/out.csv', index=False)
    sentence_sentiment_analyzer(str(tokenized_JP), str(tokenized_EN))

eng_jp_sentiment_analysis(in_JP, in_EN)