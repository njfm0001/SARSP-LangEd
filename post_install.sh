#!/bin/bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download xx_sent_ud_sm
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"