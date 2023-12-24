# tanglishServer
Backend of Tanglish

Use python3.9

The following dir structure is needed
```
models
├── best_model.pth
├── llm_lstm_model.keras
│   ├── assets
│   ├── fingerprint.pb
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── lstm_tokenizer.pkl
├── ner_crf_model.pkl
└── pos_crf_model.pkl
```
