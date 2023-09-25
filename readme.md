```
.
├── assets
│   ├── financial_blog.txt
│   └── qa_gpt4.json
├── dump
│   ├── YT_PEFT_Finetune_Bloom7B_tagger.ipynb
│   └── falcon_widget.ipynb
├── readme.md
├── scripts
│   ├── dataset_generate.ipynb
│   ├── interface
│   │   ├── chatGptDisplay.ipynb
│   │   └── chatgpt_interface_comparison.ipynb
│   └── llm.ipynb
└── src
    ├── __init__.py
    ├── db
    │   ├── discord_messge_extractor.py
    │   └── qa_generator_doc_gpt4.py
    ├── llm
    │   ├── __init__.py
    │   ├── falcon
    │   │   ├── __init__.py
    │   │   ├── falcon-7b.py
    │   │   └── falcon_finetune_qlora_langchain.ipynb
    │   └── phi
    │       ├── __init__.py
    ├── prompt
    │   └── prompt_doc.py
    └── utils
        ├── __init__.py
        ├── csv.py
        └── json_io.py
```

### assets
In this folder, there are some documents and synthesized datasets

### src
In this folder, there are python functions for dataset generating and fine-tuning, inference LLMs.
- [db]()
- [llm]()
- [prompt]()
- [utils]()

### scripts
These are ipython notbook files for dataset generating and fine-tuning, inference LLMs.
- [dataset_generate.ipynb](doc/scripts/dataset_generate.md)
- [llm.ipynb](doc/scripts/llm.md)
