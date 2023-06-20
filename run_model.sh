export MODEL="gpt2"
python run_negations.py --model_name $MODEL --max_examples 1000 
python run_toxicity.py --model_name $MODEL --max_examples 1000 
python run_lrs.py --model_name $MODEL --max_examples 1000 
python run_tokenization_split.py --model_name $MODEL --max_examples 1000 
python run_word_order.py --model_name $MODEL --max_examples 1000
