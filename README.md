# CRL_EGPG

Pytorch Implementation of **Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation**

We use contrastive loss implemented by [HobbitLong](https://github.com/HobbitLong/SupContrast).

## How to train
1. download the dataset from [here](https://drive.google.com/drive/folders/1xkCtRnbeKPg_-0qR7j8jtzV8lfEcZGJm?usp=sharing) and put it to project directory. </br>
You can directly use preprocessed dataset`(data/: QQP-Pos, data2: ParaNMT)` </br>
Or process them `(Quora and Para)` by your own through `quora_process.py` and `para_process.py` respectively.</br>
If you take the second method, you need to set the variable `text_path` in the above two programs.
2. `python train.py --dataset quora --model_save_path directory_to_save_model`
## How to evaluate

1. Firstly, generate the test target sentences by running </br>
`python evaluate --model_save_path your_saved_model --idx which_model_you_want_to_test ` </br>
After running the command, you will find the generated target file `trg_genidx.txt` and corresponding exemplar file `exmidx.txt` </br>
2. Follow the [repository](https://github.com/malllabiisc/SGCP) provided by malllabiisc. </br>
and setup the evaluation code.  Then run </br>
`python -m src.evaluation.eval -i path/trg_genidx.txt
-r path/test_trg.txt -t path/exmidx.txt` </br>
change to the corresponding path

## How to generate multiple paraphrases for one input
You can modify `generate.py` or just run </br>
`python generate.py` 
