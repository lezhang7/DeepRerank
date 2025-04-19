import copy
import json
import time
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from run_evaluation import THE_TOPICS, THE_INDEX, DLV2
from agent import get_agent
from trec_eval import eval_rerank
from utils import set_seed


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


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        # messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def sliding_windows_batch(agent, items, rank_start=0, rank_end=100, window_size=20, step=10):
    """Process multiple items with sliding windows using batched inference."""
    items = [copy.deepcopy(item) for item in items]
    results = []
    
    # Initialize positions for all items
    all_positions = []
    for _ in items:
        end_pos = rank_end
        start_pos = rank_end - window_size
        item_positions = []
        
        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            item_positions.append((start_pos, end_pos))
            end_pos = end_pos - step
            start_pos = start_pos - step
            
        all_positions.append(item_positions)
    
    # Process each window position across all items in batches
    max_windows = max(len(positions) for positions in all_positions)
    
    for window_idx in range(max_windows):
        # Collect messages for this window position across all items
        batch_messages = []
        batch_metadata = []  # Store (item_idx, start_pos, end_pos) for each message
        
        for item_idx, (item, positions) in enumerate(zip(items, all_positions)):
            if window_idx < len(positions):
                start_pos, end_pos = positions[window_idx]
                messages = create_permutation_instruction(item=item, rank_start=start_pos, rank_end=end_pos)
                batch_messages.append(messages)
                batch_metadata.append((item_idx, start_pos, end_pos))
        
        if not batch_messages:
            continue
            
        # Process this batch of messages
        batch_permutations = agent.batch_chat(batch_messages, temperature=0, return_text=True, seed=42)
        
        # Apply permutations to respective items
        for (item_idx, start_pos, end_pos), permutation in zip(batch_metadata, batch_permutations):
            if "ERROR::" in permutation:
                print(f"Error in batch processing: {permutation}")
                continue
            items[item_idx] = receive_permutation(items[item_idx], permutation, 
                                                 rank_start=start_pos, rank_end=end_pos)
    
    return items


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def process_rank_results_in_batches(agent, rank_results, batch_size=8, window_size=20, step=10):
    """
    Process ranking results in batches to improve throughput.
    
    Args:
        agent: The LLM agent to use for reranking
        rank_results: List of ranking results to process
        batch_size: Number of items to process in each batch
        window_size: Size of sliding window for reranking
        step: Step size for sliding window
        
    Returns:
        List of processed ranking results
    """
    new_results = []
    total_start_time = time.time()
    
    for i in range(0, len(rank_results), batch_size):
        batch_items = rank_results[i:i+batch_size]
        print(f"Processing batch: {i//batch_size + 1}/{(len(rank_results) + batch_size - 1)//batch_size}")
        
        # Process entire batch at once
        processed_items = sliding_windows_batch(agent, batch_items, 
                                               rank_start=0, rank_end=100, 
                                               window_size=window_size, step=step)
        new_results.extend(processed_items)
        
        print(f"Completed {len(new_results)}/{len(rank_results)} items")
    
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


def main():

    set_seed(42)
    for data in ['dl21', 'dl22', 'dl23']:
        rank_results = bm25_retrieve(data, top_k_retrieve=100)
        # model_name = "Qwen/Qwen2.5-7B-Instruct"
        # agent = get_agent(model_name=model_name, api_key=None)
        # rank_results = process_rank_results_in_batches(agent, rank_results)
        
        # save rank_results
        with open(f'/home/mila/l/le.zhang/scratch/DeepRerank/data/{data}_bm25_rank_results.json', 'w') as f:
            json.dump(rank_results, f, indent=4)
        all_metrics = eval_rerank(data, rank_results)


if __name__ == '__main__':
    main()
