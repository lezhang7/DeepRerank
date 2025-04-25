import sys
sys.path.append('/home/mila/l/le.zhang/scratch/DeepRerank')
from datasets import load_dataset   
import os
import json
from rank_gpt import process_rank_results_in_batches
from trec_eval import eval_rerank
from utils import set_seed
from agent import get_agent

if __name__ == "__main__":
    global DEBUG
    DEBUG = True
    dataset = load_dataset("parquet", data_files='filtered_train_high_ndcg.parquet',split='train[:20]')
    
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    set_seed(42)    
    # model_name = "le723z/qwen2_7b_deeprerank_ndcgreward10_3reward_v3"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"model_name: {model_name}")
    agent = get_agent(model_name=model_name, api_key=None)
    
    for qid, data in enumerate(dataset):
        
        # bs is 16*num_gpu, gpu automatically allocated
        num_gpu = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
        bs = 16*num_gpu
         # save rank_results
        # with open(f'/home/mila/l/le.zhang/scratch/DeepRerank/data/{data}_bm25_rank_results.json', 'w') as f:
        #     json.dump(rank_results, f, indent=4)
        original_metrics, _ = eval_rerank('/home/mila/l/le.zhang/scratch/DeepRerank/data/combined_qrels.txt', data)
        
        for idx in range(3):
            rank_results = process_rank_results_in_batches(agent, [data], batch_size=bs, verbose=True)
            all_metrics, _ = eval_rerank('/home/mila/l/le.zhang/scratch/DeepRerank/data/combined_qrels.txt', rank_results)
            
            ordered_all_metrics = {}
            for key in original_metrics.keys():
                ordered_all_metrics[key] = all_metrics[key]
            data[f"rerank_metrics@{idx}"] = ordered_all_metrics
            print(f"original score: {original_metrics}")
            print(f"zero-shot rerank score {idx}: {ordered_all_metrics}")

        data['metrics'] = original_metrics
        # Save data with rerank scores to json file
        output_path = f'/home/mila/l/le.zhang/scratch/DeepRerank/data/zero_shot_rerank/rerank_results_{qid}.json'
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        # breakpoint()

    