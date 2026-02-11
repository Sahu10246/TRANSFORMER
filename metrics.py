from rouge_score import rouge_scorer

def rouge_scores(preds,targets):
    scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    return [scorer.score(t,p) for p,t in zip(preds,targets)]
