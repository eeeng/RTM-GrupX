import argparse
from langchain_util import LangChainModel
from qa.qa_reader import qa_read
from hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--query', type=str)
    args = parser.parse_args()

    hipporag = HippoRAG(corpus_name=args.dataset, extraction_model='openai', extraction_model_name='gpt-3.5-turbo-1106',
                 graph_creating_retriever_name='colbertv2', qa_model=LangChainModel('openai', 'gpt-3.5-turbo'))

    qa_few_shot_samples = None
    queries = [args.query]
    for query in queries:
        ranks, scores, logs = hipporag.rank_docs(query, top_k=2)
        retrieved_passages = [hipporag.get_passage_by_idx(rank) for rank in ranks]
        response = qa_read(query, retrieved_passages, qa_few_shot_samples, hipporag.qa_model)
        print(f"{response=}")
        print(ranks)
        print(scores)
        print(logs)
