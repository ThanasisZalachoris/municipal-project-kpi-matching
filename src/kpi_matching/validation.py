"""Input contracts and non-statistical fixture checks; no fabricated ML metrics."""

from collections import Counter,defaultdict
import re

from .text import tokens


def validate(document):
    if set(document) != {'schema_version','data_kind','catalogue','projects','policy','expectations'}:
        raise ValueError('Invalid input schema')
    if type(document['schema_version']) is not int or document['schema_version'] != 1 or document['data_kind'] != 'synthetic':
        raise ValueError('Only explicit versioned synthetic fixtures supported')
    ids=set()
    for kpi in document['catalogue']:
        if set(kpi) != {'id','name','description','category'} or not re.fullmatch(r'[a-z][a-z0-9_]*',kpi['id']):
            raise ValueError('Invalid KPI schema')
        if kpi['id'] in ids or any(not isinstance(kpi[k],str) or not kpi[k].strip() for k in ['name','description','category']):
            raise ValueError('Duplicate or empty KPI')
        if not re.fullmatch(r'[a-z][a-z0-9_]*',kpi['category']):raise ValueError('Invalid category')
        tokens(kpi['name']+' '+kpi['description']);ids.add(kpi['id'])
    projects=set()
    for project in document['projects']:
        if set(project) != {'id','text'} or not re.fullmatch(r'[a-z][a-z0-9_]*',project['id']) or project['id'] in projects:
            raise ValueError('Invalid or duplicate project')
        tokens(project['text']);projects.add(project['id'])
    if not projects or not ids:raise ValueError('Empty project set or catalogue')
    if set(document['policy']) != {'top_k','min_score','min_margin'}:raise ValueError('Explicit policy required')
    seen=set()
    for expectation in document['expectations']:
        if set(expectation) != {'project_id','expected_top','expect_abstain'}:
            raise ValueError('Invalid expectation schema')
        if expectation['project_id'] not in projects or expectation['project_id'] in seen:
            raise ValueError('Unknown or duplicate expectation project')
        if expectation['expected_top'] is not None and expectation['expected_top'] not in ids:
            raise ValueError('Unknown expected KPI')
        if type(expectation['expect_abstain']) is not bool:raise ValueError('Explicit abstention expectation required')
        seen.add(expectation['project_id'])
    if seen != projects:raise ValueError('Every fixture project needs a declared expectation')


def evaluation(document, results, baseline):
    categories={k['id']:k['category'] for k in document['catalogue']}
    support=Counter();groups=defaultdict(list);cases=[]
    for project in document['projects']:
        # Identical unigram representations are not independent examples.
        key=tuple(sorted(Counter(tokens(project['text'])).items()))
        groups[key].append(project['id'])
    for expected in sorted(document['expectations'],key=lambda e:e['project_id']):
        pid=expected['project_id'];actual=results[pid];target=expected['expected_top']
        if target:support[categories[target]]+=1
        baseline_top=baseline[pid][0]['kpi_id'] if baseline[pid][0]['score']>0 else None
        top=actual['candidates'][0]['kpi_id'] if actual['top_score']>0 else None
        matches=(target is None or top==target) and (actual['decision']=='abstain')==expected['expect_abstain']
        cases.append({'project_id':pid,'expected_top':target,'tfidf_top':top,'overlap_top':baseline_top,
                      'expect_abstain':expected['expect_abstain'],'actual_decision':actual['decision'],
                      'fixture_check':'pass' if matches else 'fail'})
    return {'data_kind':'synthetic','evaluation_kind':'functional_synthetic_fixture_checks','cases':cases,
            'supervised_metrics':None,'cross_validation':None,'classifier_trained':False,
            'independent_label_source':False,'expected_category_counts':dict(sorted(support.items())),
            'unique_text_groups':len(groups),'duplicate_text_groups':[sorted(ids) for ids in groups.values() if len(ids)>1],
            'metrics_status':'not_reported',
            'reasons':['Expectations are authored synthetic regression checks, not independent relevance judgments.',
                       'Small, imbalanced category support cannot establish generalization.',
                       'Similarity ranking is not supervised classification; no resampling or classifier fit occurs.'],
            'baseline':'Unweighted token-set Jaccard overlap; comparative diagnostics only.'}
