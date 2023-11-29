#!/usr/bin/env python3
from flask import Flask, json, jsonify, request
from nltk import word_tokenize, sent_tokenize
import re
import string

app = Flask(__name__)


def conv_uchar_to_space(str_):
    return "".join([i if ord(i) < 128 else " " for i in str_])


def remove_punctuation(str_: str):
    return str_.translate(str.maketrans("", "", string.punctuation))


@app.route("/preprocess/", methods=["POST"])
def handle_preprocess():
    corpus = request.get_json()
    corpus = corpus["text"]
    corpus = sent_tokenize(corpus)
    for ind, val in enumerate(corpus):
        tmp = conv_uchar_to_space(val)
        tmp = tmp.lower()
        tmp = re.sub(r"\d+", "", tmp)
        tmp = remove_punctuation(tmp)
        tmp = word_tokenize(tmp)
        corpus[ind] = tmp
    return jsonify(corpus)
