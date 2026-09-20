"""Offline orchestration and reproducible decision-support artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .matching import Matcher,overlap_baseline
from .text import PREPROCESSING_VERSION,STOPWORDS
from .validation import evaluation,validate


def render(results, catalogue, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    pids=sorted(results);kids=sorted(k['id'] for k in catalogue)
    scores=[{x['kpi_id']:x['score'] for x in results[p]['all_scores']} for p in pids]
    matrix=[[r[k] for k in kids] for r in scores]
    fig,ax=plt.subplots(figsize=(10,6))
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(kids)),kids,rotation=25,ha='right')
    ax.set_yticks(range(len(pids)),pids)
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value>.65 else '#17253b')
    ax.set_title('SYNTHETIC project-to-KPI lexical similarity\nCosine scores are not probabilities or municipal performance',pad=15)
    fig.colorbar(im,ax=ax,label='Cosine similarity')
    fig.tight_layout();fig.savefig(output/'similarity.png',dpi=120,metadata={'Software':'Synthetic KPI matching'});plt.close(fig)


def run(input_path, output_dir):
    input_path,output=Path(input_path),Path(output_dir)
    if output.resolve()==input_path.resolve().parent or output.resolve() in input_path.resolve().parents:
        raise ValueError('Output must be separate from fixture directory')
    raw=input_path.read_bytes();document=json.loads(raw);validate(document)
    matcher=Matcher(document['catalogue']);results={};baselines={}
    for project in sorted(document['projects'],key=lambda p:p['id']):
        results[project['id']]=matcher.rank(project['text'],**document['policy'])
        baselines[project['id']]=overlap_baseline(project['text'],document['catalogue'])
    checks=evaluation(document,results,baselines)
    failed=sum(c['fixture_check']=='fail' for c in checks['cases'])
    report={'data_kind':'synthetic','input_sha256':hashlib.sha256(raw).hexdigest(),
            'catalogue_sha256':hashlib.sha256(json.dumps(matcher.catalogue,sort_keys=True).encode()).hexdigest(),
            'projects':len(results),'kpis':len(matcher.catalogue),'vocabulary_size':len(matcher.idf),
            'policy':document['policy'],'fixture_failures':failed,'requires_human_review':True,
            'status':'fixture_failure' if failed else 'fixture_checks_passed',
            'preprocessing_version':PREPROCESSING_VERSION,'stopwords':sorted(STOPWORDS),
            'fit_scope':'KPI catalogue only; query projects never fit vocabulary or IDF.'}
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('rankings.json',{'data_kind':'synthetic','projects':results}),
                      ('baseline.json',{'data_kind':'synthetic','projects':baselines}),('evaluation.json',checks),
                      ('validation.json',report),('catalogue_model.json',{'data_kind':'synthetic','idf':matcher.idf,'catalogue_ids':[k['id'] for k in matcher.catalogue]})]:
        (output/name).write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    with (output/'candidates.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['project_id','rank','kpi_id','category','score','decision','reason','margin','shared_terms','data_kind'],lineterminator='\n')
        writer.writeheader()
        for pid,r in results.items():
            for candidate in r['candidates']:
                writer.writerow({'project_id':pid,**candidate,'decision':r['decision'],'reason':r['reason'],
                                 'margin':r['margin'],'shared_terms':json.dumps(candidate['shared_terms'],sort_keys=True),
                                 'data_kind':'synthetic'})
    render(results,document['catalogue'],output)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='Rank invented municipal project text against a synthetic KPI catalogue')
    parser.add_argument('--input',default='data/sample/fixture.json')
    parser.add_argument('--output',default='outputs/sample')
    args=parser.parse_args(argv)
    try:report=run(args.input,args.output)
    except (ValueError,KeyError,TypeError,OSError):
        print('Invalid input contract or output failure');return 1
    print(json.dumps(report,sort_keys=True));return 2 if report['fixture_failures'] else 0
