"""Catalogue-only smoothed TF-IDF, L2 normalization and transparent cosine ranking."""

from collections import Counter
import math

from .text import tokens


class Matcher:
    def __init__(self, catalogue):
        self.catalogue = sorted(catalogue, key=lambda k: k['id'])
        self.documents = {k['id']: tokens(k['name']+' '+k['description']) for k in self.catalogue}
        if any(not words for words in self.documents.values()):
            raise ValueError('Every KPI needs usable representation text')
        frequencies = Counter(w for words in self.documents.values() for w in set(words))
        n = len(self.documents)
        if not n:
            raise ValueError('Empty catalogue')
        self.idf = {w: math.log((1+n)/(1+df))+1 for w, df in sorted(frequencies.items())}
        self.vectors = {key: self.vector(words) for key, words in self.documents.items()}

    def vector(self, words):
        counts = Counter(w for w in words if w in self.idf)
        weights = {w: counts[w]*self.idf[w] for w in sorted(counts)}
        norm = math.sqrt(sum(v*v for v in weights.values()))
        return {w: v/norm for w, v in weights.items()} if norm else {}

    def rank(self, text, top_k=3, min_score=.12, min_margin=.05):
        if type(top_k) is not int or not 1 <= top_k <= len(self.catalogue):
            raise ValueError('Invalid top-k')
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0 <= x <= 1 for x in [min_score,min_margin]):
            raise ValueError('Invalid review thresholds')
        words = tokens(text);query = self.vector(words)
        scores = []
        for kpi in self.catalogue:
            vector = self.vectors[kpi['id']]
            contributions = [{'term': w, 'contribution': round(query[w]*vector[w],12)} for w in sorted(query.keys() & vector.keys())]
            score = min(1.0, max(0.0, sum(query[w]*vector[w] for w in sorted(query.keys() & vector.keys()))))
            scores.append({'kpi_id': kpi['id'], 'category': kpi['category'], 'score': round(score,12),
                           'shared_terms': sorted(contributions,key=lambda c:(-c['contribution'],c['term']))})
        scores.sort(key=lambda r:(-r['score'],r['kpi_id']))
        best = scores[0]['score'];runner_up = scores[1]['score'] if len(scores)>1 else 0.0
        margin = round(best-runner_up,12)
        reason = ('empty_text' if not words else 'out_of_vocabulary' if not query else
                  'below_minimum_score' if best < min_score else 'ambiguous_margin' if margin < min_margin else 'candidate_requires_review')
        suggest = reason == 'candidate_requires_review'
        return {'suggested_kpi': scores[0]['kpi_id'] if suggest else None,
                'decision': 'review_candidate' if suggest else 'abstain', 'reason': reason,
                'top_score': best, 'margin': margin, 'token_count': len(words),
                'known_token_count': sum(w in self.idf for w in words),
                'oov_terms': sorted(set(words)-self.idf.keys()),
                'candidates': [dict(row,rank=i+1) for i,row in enumerate(scores[:top_k])],
                'all_scores': scores}


def overlap_baseline(text, catalogue):
    query = set(tokens(text));result=[]
    for kpi in catalogue:
        words=set(tokens(kpi['name']+' '+kpi['description']))
        union=query|words
        result.append({'kpi_id':kpi['id'],'score':round(len(query&words)/len(union),12) if union else 0.0})
    return sorted(result,key=lambda r:(-r['score'],r['kpi_id']))
