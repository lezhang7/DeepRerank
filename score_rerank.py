import copy
import json
import time
import re
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from run_evaluation import THE_TOPICS, THE_INDEX, DLV2
from agent import get_agent
from trec_eval import eval_rerank
from utils import set_seed
import os

def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                elif 'passage' in content:
                    content = content['passage']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
                    
    return ranks


def get_scoring_system_prompt():
    return {
        "role": "system",
        "content": "You are ScoreRerank, an intelligent assistant that evaluates the relevance of passages to a search query. You will score each passage on a scale of 1-5 based on how well it addresses the query."
    }


def create_scoring_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])
    max_length = 300
    
    messages = [
        get_scoring_system_prompt(),
        {
            "role": "user",
            "content": f"I will provide you with {num} passages. For each passage, evaluate its relevance to the query: \"{query}\" on a scale of 1-5, where:\n\n"
                      f"5: Directly answers the query with comprehensive, accurate information\n"
                      f"4: Highly relevant with most key information but may lack some details\n"
                      f"3: Moderately relevant with some useful information\n"
                      f"2: Tangentially relevant with minimal useful information\n"
                      f"1: Not relevant or off-topic\n\n"
                      f"For each passage, provide your score in the format [relevance: X] where X is your score from 1-5."
        },
        {
            "role": "assistant", 
            "content": "I'll evaluate each passage and provide a relevance score from 1-5 in the format [relevance: X]. Please provide the passages."
        }
    ]
    
    # Add passages one by one
    for i, hit in enumerate(item['hits'][rank_start: rank_end]):
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split()[:int(max_length)])
        
        messages.append({
            "role": "user", 
            "content": f"Passage {i+1}:\n{content}"
        })
        
    # Final instruction
    messages.append({
        "role": "user",
        "content": f"Please evaluate all {num} passages above for their relevance to the query: \"{query}\". For each passage, provide a score in the format [relevance: X] where X is 1-5. Start each evaluation with \"Passage X:\" and include a brief explanation of your reasoning before giving the score."
    })
    
    return messages


def extract_relevance_scores(response):
    """Extract relevance scores from the model's response."""
    scores = []
    pattern = r"Passage (\d+):.*?\[relevance: (\d+)\]"
    matches = re.findall(pattern, response, re.DOTALL)
    
    # Create a dictionary to store passage number -> score
    score_dict = {}
    for match in matches:
        passage_num = int(match[0])
        score = int(match[1])
        score_dict[passage_num] = score
    
    # Convert to ordered list
    for i in range(1, max(score_dict.keys()) + 1):
        if i in score_dict:
            scores.append((i, score_dict[i]))
    
    if DEBUG:
        print(f"Extracted scores: {scores}")
    
    return scores


def rerank_by_scores(item, scores, rank_start=0, rank_end=100):
    """Rerank passages based on relevance scores."""
    # Sort by score (descending) and then by original rank (ascending) for ties
    sorted_scores = sorted(scores, key=lambda x: (-x[1], x[0]))
    
    if DEBUG:
        print(f"Sorted scores: {sorted_scores}")
    
    # Get the new order of passage indices (0-indexed)
    new_order = [idx - 1 for idx, _ in sorted_scores]
    
    # Apply the reordering
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    for j, idx in enumerate(new_order):
        if idx < len(cut_range):
            item['hits'][j + rank_start] = copy.deepcopy(cut_range[idx])
            if 'rank' in item['hits'][j + rank_start]:
                item['hits'][j + rank_start]['rank'] = j + rank_start + 1
            if 'score' in item['hits'][j + rank_start]:
                # Use the relevance score as the new score, scaled to match original score range
                original_max = max(hit['score'] for hit in cut_range)
                original_min = min(hit['score'] for hit in cut_range)
                relevance_score = scores[j][1]
                # Scale from 1-5 to original score range
                scaled_score = original_min + (relevance_score - 1) * (original_max - original_min) / 4
                item['hits'][j + rank_start]['score'] = scaled_score
    
    return item


def process_batch_scoring(agent, items, batch_size=8):
    """Process items in batches for scoring and reranking."""
    new_results = []
    total_start_time = time.time()
    
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        print(f"Processing batch: {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
        
        # Process each item in the batch
        processed_items = []
        for item in batch_items:
            # Score all passages at once
            messages = create_scoring_instruction(item, rank_start=0, rank_end=len(item['hits']))
            
            # Get model response
            response = agent.chat(messages, temperature=0, return_text=True, seed=42)
            
            if DEBUG:
                print("="*100)
                print(f"Query: {item['query']}")
                print(f"Response excerpt: {response[:500]}...")
            
            # Extract scores
            scores = extract_relevance_scores(response)
            
            # Rerank based on scores
            item = rerank_by_scores(item, scores, rank_start=0, rank_end=len(item['hits']))
            
            processed_items.append(item)
        
        new_results.extend(processed_items)
        print(f"Completed {len(new_results)}/{len(items)} items")
    
    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f}s")
    
    return new_results


def bm25_retrieve(data, top_k_retrieve=100):
    assert data in THE_INDEX, f"Data {data} not found in THE_INDEX"
    searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
    topics = get_topics(THE_TOPICS[data] if data not in DLV2 else data)
    qrels = get_qrels(THE_TOPICS[data])
    rank_results = run_retriever(topics, searcher, qrels, k=top_k_retrieve)
    return rank_results


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} score_rerank\n")
                rank += 1
    return True


def main():
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    set_seed(42)
    DEBUG = True
    # Choose your model
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    print(f"Model name: {model_name}")
    
    agent = get_agent(model_name=model_name, api_key=None)
    
    for data in ['dl19']:
        print(f"Processing dataset: {data}")
        
        # Retrieve documents using BM25
        rank_results = bm25_retrieve(data, top_k_retrieve=100)
        
        # Determine batch size based on available GPUs
        num_gpu = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
        bs = 1 * num_gpu  # Smaller batch size for scoring which is more intensive
        
        # Process and rerank documents
        reranked_results = process_batch_scoring(agent, rank_results, batch_size=bs)
        
        # Save results if needed
        output_file = f'results/{data}_score_rerank_results.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(reranked_results, f, indent=4)
        
        # Evaluate results
        all_metrics, _ = eval_rerank(data, reranked_results)
        print(f"Evaluation metrics for {data}:")
        print(all_metrics)


if __name__ == '__main__':
    DEBUG = True
    main() 