#!/usr/bin/env python3
from ai4bharat.transliteration.transformer.base_engine import lang_code
from flask import Flask, json, jsonify, request
from nltk import corpus, word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from ai4bharat.transliteration import XlitEngine
import re
import string

app = Flask(__name__)
transliteration_model = XlitEngine("ta", beam_width=20, rescore=True)


def conv_uchar_to_space(str_):
    return "".join([i if ord(i) < 128 else " " for i in str_])


def remove_punctuation(str_: str):
    return str_.translate(str.maketrans("", "", string.punctuation))


def preprocess(str_):
    str_ = sent_tokenize(str_)
    for ind, val in enumerate(str_):
        tmp = conv_uchar_to_space(val)
        tmp = tmp.lower()
        tmp = re.sub(r"\d+", "", tmp)
        tmp = remove_punctuation(tmp)
        tmp = word_tokenize(tmp)
        str_[ind] = tmp
    return str_


def ngram(str_, n):
    str_ = preprocess(str_)
    return padded_everygram_pipeline(n, str_)


def transliterate(str_):
    return transliteration_model.translit_sentence(str_, lang_code="ta")


@app.route("/preprocess/", methods=["POST"])
def handle_preprocess():
    corpus = request.get_json()
    corpus = corpus["text"]
    corpus = preprocess(corpus)
    return jsonify(corpus)


@app.route("/ngram/", methods=["POST"])
def handle_ngram():
    req = request.get_json()
    corpus = req["text"]
    n = req["n"]
    train, vocab = ngram(corpus, n)
    ret = {}
    ret["train"] = [list(x) for x in train]
    ret["vocab"] = [x for x in vocab]
    return jsonify(ret)


@app.route("/transliterate/", methods=["POST"])
def handle_transliterate():
    corpus = request.get_json()["text"]
    return jsonify(transliterate(corpus))
