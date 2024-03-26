# Lab4_LLM_Traditional-Chinese-Reading-Comprehension-Test-for-LLMs

Goal
-
The main purpose of this experiment is to establish a Chinese large language model (LLM) to answer questions and an example is showed as followings. Moreover, the competition is on Kaggle (https://www.kaggle.com/competitions/trad-chinese-reading-comprehension-test-for-llms/overview).

(ex.)
- "instruction": "請根據以下輸入回答選擇題，並以數字回答:\n",
- "input": "明 年 元 旦 起 ， 交 通 部 新 規 定 ， 汽 車 輪 胎 胎 紋 深 度 將 納 入 定 期 檢 驗 項 目 之 一 ， 一 旦 深 度 未 達 1 . 6 公 厘 ， 及 1 個 月 內 沒 換 胎 ， 將 會 被 吊 銷 牌 照 。 民 眾 除 了 定 期 檢 測 胎 紋 ， 也 可 以 自 已 用 1 0 元 硬 幣 檢 測 ， 只 要 看 的 見 國 父 像 衣 領 下 緣 ， 表 示 該 換 輪 胎 了 。 \n \n 問 題 : 汽 車 胎 紋 未 到 達 多 少 公 厘 將 會 被 吊 銷 牌 照 ? \n 1 : 1 . 5 公 厘 \n 2 : 1 . 4 公 厘 \n 3 : 1 . 3 公 厘 \n 4 : 1 . 6 公 厘 \n",
- "output": "4"
  

File Discription
-
1. Data
    - train.json : Training data.
    - valid.json : Validation data.
    - AI1000.json : Testing data. 
2. Code
    - trainer.py : Main code on Colab for training LoRA model.
    - convert_and_quantize_chinese_llama2_and_alpaca2.ipynb : Code on Colab for combining trained LoRA model and Chinese-Alpaca-2-13B traind by https://github.com/ymcui/Chinese-LLaMA-Alpaca-2.
    - inference.ipynb : Code on Colab for inference.
    - run_sft.sh : Script to run.
3. Result
    - Due to the file size limitations on GitHub, you can access the following files via the provided cloud link: 
      https://drive.google.com/drive/folders/1dwgqd7fX_-tPEZb0g2V6yN7822_YVd97?usp=drive_link
    - answer2000.csv : The prediction based on testing data (AI1000.json).
  

How to start up
-
1. Step 1 - Train LoRA model:
Please open file run_sft.sH. And in the following lines, you can modify these parameters (reference to https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh). Then, you can open trainer.ipynb and run it.

        lr=1e-4
        lora_rank=8
        lora_alpha=16
        lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
        modules_to_save="embed_tokens,lm_head"
        lora_dropout=0.05

        pretrained_model="hfl/chinese-alpaca-2-13b"
        chinese_tokenizer_path="hfl/chinese-alpaca-2-13b"
        dataset_dir="${root}../../../../Data"
        per_device_train_batch_size=1
        per_device_eval_batch_size=1
        gradient_accumulation_steps=8
        max_seq_length=512
        output_dir="${root}../../../../Result"
        validation_file="${dataset_dir}/valid.json"

        deepspeed_config_file=ds_zero2_no_offload.json

        torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
             --deepspeed ${deepspeed_config_file} \
             --model_name_or_path ${pretrained_model} \
             --tokenizer_name_or_path ${chinese_tokenizer_path} \
             --dataset_dir ${dataset_dir} \
             --per_device_train_batch_size ${per_device_train_batch_size} \
             --per_device_eval_batch_size ${per_device_eval_batch_size} \
             --do_train \
             --do_eval \
             --seed $RANDOM \
             --fp16 \
             --num_train_epochs 5 \
             --lr_scheduler_type cosine \
             --learning_rate ${lr} \
             --warmup_ratio 0.03 \
             --weight_decay 0 \
             --logging_strategy steps \
             --logging_steps 10 \
             --save_strategy steps \
             --save_total_limit 3 \
             --evaluation_strategy steps \
             --eval_steps 1000 \
             --save_steps 500 \
             --gradient_accumulation_steps ${gradient_accumulation_steps} \
             --preprocessing_num_workers 4 \
             --max_seq_length ${max_seq_length} \
             --output_dir ${output_dir} \
             --overwrite_output_dir \
             --ddp_timeout 30000 \
             --logging_first_step True \
             --lora_rank ${lora_rank} \
             --lora_alpha ${lora_alpha} \
             --trainable ${lora_trainable} \
             --lora_dropout ${lora_dropout} \
             --torch_dtype float16 \
             --validation_file ${validation_file} \
             --load_in_kbits 4 \
             --save_safetensors False \
             --gradient_checkpointing \
             --ddp_find_unused_parameters False
   
2. Step 2 - Combine trained LoRA model and traind LLM model :
   Open convert_and_quantize_chinese_llama2_and_alpaca2.ipynb. And in the following lines, you can modify these paths to your files.
      
          !python ./scripts/merge_llama2_with_chinese_lora_low_mem.py \
                  --base_model hfl/chinese-alpaca-2-13b \
                  --lora_model /content/drive/MyDrive/ColabNotebooks/2023ML/Lab4/Result/checkpoint-2000/sft_lora_model \
                  --output_type huggingface \
                  --output_dir llama-2-13b-combined-2000
3. Step 3 - Inference :
   Open inference.ipynb and run it.

Methodology
-
1. Data Discription

     In order to use text as input for the model's learning process, in this experiment, we first break down each sentence in Chinese and Tailo into individual characters. Then, these characters are encoded to establish a directory mapping each character to its encoding. Below is the code implementing Word2Vector.

    | Training data : 13550 sentences.
   
    | Validation data : 74 sentences.
    
    | Testing data : 1000 sentences.
   
3. Hyperparameters :
    - Step : 2000
    - Learning rate : 1e-4
    - lora_rank : 8
    - lora_alpha : 16
    - Max sequence length : 512


Results & Disscusion
-

1. Results
   - Best testing on Kaggle :
       - Private score : 0.80428.
         
2. Disscussion

     - Compare to the results of applying different settings

          This experiment also attempted to use a larger lora_rank and lora_alpha for training.The predicted results yielded a lower Kaggle private score of 0.74857, performing worse. The detailed lora_rank and lora_alpha settings are as follows:

        - lora_rank : 64
        - lora_alpha : 128
