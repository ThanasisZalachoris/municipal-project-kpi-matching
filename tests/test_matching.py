import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kpi_matching.matching import Matcher,overlap_baseline
from kpi_matching.pipeline import main,run
from kpi_matching.text import tokens
from kpi_matching.validation import evaluation,validate


def fixture():return json.loads((ROOT/'data/sample/fixture.json').read_text())


def kpi(key,text,category='demo'):
    return {'id':key,'name':text,'description':text,'category':category}


class TextTests(unittest.TestCase):
    def test_case_punctuation_and_stopwords(self):
        self.assertEqual(tokens('  WATER, and Pipes! '),['water','pipes'])

    def test_unicode_not_erased_by_ascii_filter(self):
        self.assertEqual(tokens('ΔΗΜΟΣ πράσινο'),['δημοσ','πράσινο'])

    def test_compatibility_normalization(self):
        self.assertEqual(tokens('ＷＡＴＥＲ'),['water'])

    def test_missing_text_rejected(self):
        for text in [None,42,[]]:
            with self.assertRaises(ValueError):tokens(text)

    def test_blank_and_stopwords_have_no_features(self):
        self.assertEqual(tokens('the and of!'),[])

    def test_unbounded_input_rejected(self):
        with self.assertRaises(ValueError):tokens('x'*10001)


class MatchingTests(unittest.TestCase):
    def setUp(self):self.f=fixture();self.m=Matcher(self.f['catalogue'])

    def test_smoothed_idf_formula(self):
        m=Matcher([kpi('a','water pipes'),kpi('b','water trees')])
        self.assertEqual(m.idf['water'],1)
        self.assertAlmostEqual(m.idf['pipes'],math.log(3/2)+1)

    def test_unit_length_vectors(self):
        for v in self.m.vectors.values():self.assertAlmostEqual(sum(x*x for x in v.values()),1)

    def test_exact_representation_cosine_one(self):
        k=self.f['catalogue'][0]
        r=self.m.rank(k['name']+' '+k['description'])
        self.assertEqual(r['candidates'][0]['kpi_id'],k['id'])
        self.assertAlmostEqual(r['top_score'],1)

    def test_explanation_sums_to_score(self):
        for candidate in self.m.rank('solar electricity water pipes')['candidates']:
            self.assertAlmostEqual(sum(x['contribution'] for x in candidate['shared_terms']),candidate['score'],places=10)

    def test_empty_query_abstains(self):
        r=self.m.rank('');self.assertIsNone(r['suggested_kpi']);self.assertEqual(r['reason'],'empty_text')

    def test_oov_query_never_assigned_first_catalogue_row(self):
        r=self.m.rank('quantum cryptography')
        self.assertEqual(r['reason'],'out_of_vocabulary');self.assertIsNone(r['suggested_kpi'])

    def test_low_margin_abstains(self):
        self.assertEqual(self.m.rank('access')['reason'],'ambiguous_margin')

    def test_high_cutoff_exposes_below_threshold(self):
        self.assertEqual(self.m.rank('solar',min_score=1)['reason'],'below_minimum_score')

    def test_ties_have_stable_id_order(self):
        m=Matcher([kpi('b','same text'),kpi('a','same text')])
        r=m.rank('same text',top_k=2)
        self.assertEqual([x['kpi_id'] for x in r['candidates']],['a','b'])
        self.assertEqual(r['decision'],'abstain')

    def test_margin_uses_full_catalogue_even_when_top_one_requested(self):
        self.assertEqual(self.m.rank('access',top_k=1)['margin'],self.m.rank('access',top_k=3)['margin'])

    def test_catalogue_order_does_not_change_results(self):
        m=Matcher(list(reversed(self.f['catalogue'])))
        self.assertEqual(self.m.rank('water pipes'),m.rank('water pipes'))

    def test_query_does_not_fit_vocabulary_or_idf(self):
        before=copy.deepcopy((self.m.idf,self.m.vectors))
        for q in ['newholdouttoken newholdouttoken','water water water','']:
            self.m.rank(q)
        self.assertEqual(before,(self.m.idf,self.m.vectors))
        self.assertNotIn('newholdouttoken',self.m.idf)

    def test_query_batch_cannot_change_other_query_ranking(self):
        before=self.m.rank('water pipes')
        for _ in range(20):self.m.rank('water solar newholdouttoken')
        self.assertEqual(before,self.m.rank('water pipes'))

    def test_invalid_policy(self):
        for kwargs in [{'top_k':0},{'top_k':True},{'top_k':99},{'min_score':float('nan')},{'min_margin':-1}]:
            with self.assertRaises(ValueError):self.m.rank('water',**kwargs)

    def test_empty_catalogue_or_representation(self):
        for c in [[],[kpi('x','the and')]]:
            with self.assertRaises(ValueError):Matcher(c)

    def test_jaccard_baseline_known_result(self):
        r=overlap_baseline('water solar',[kpi('a','water pipes')])
        self.assertAlmostEqual(r[0]['score'],1/3)


class ValidationTests(unittest.TestCase):
    def test_valid_schema(self):validate(fixture())

    def test_duplicate_id_rejected(self):
        f=fixture();f['projects'].append(f['projects'][0])
        with self.assertRaises(ValueError):validate(f)

    def test_missing_text_field_rejected(self):
        f=fixture();del f['projects'][0]['text']
        with self.assertRaises(ValueError):validate(f)

    def test_missing_kpi_description_rejected(self):
        f=fixture();f['catalogue'][0]['description']=''
        with self.assertRaises(ValueError):validate(f)

    def test_bad_expected_kpi_rejected(self):
        f=fixture();f['expectations'][0]['expected_top']='unknown'
        with self.assertRaises(ValueError):validate(f)

    def test_all_fixture_cases_required(self):
        f=fixture();f['expectations'].pop()
        with self.assertRaises(ValueError):validate(f)

    def test_score_derived_labels_not_accepted(self):
        f=fixture();f['labels']=[1,0]
        with self.assertRaises(ValueError):validate(f)

    def test_no_supervised_metrics_from_toy_expectations(self):
        f=fixture();m=Matcher(f['catalogue'])
        r={p['id']:m.rank(p['text']) for p in f['projects']}
        b={p['id']:overlap_baseline(p['text'],f['catalogue']) for p in f['projects']}
        e=evaluation(f,r,b)
        self.assertIsNone(e['supervised_metrics']);self.assertIsNone(e['cross_validation'])
        self.assertFalse(e['classifier_trained'])
        self.assertEqual(e['unique_text_groups'],9)
        self.assertEqual(e['duplicate_text_groups'],[['crossings','crossings_copy']])
        self.assertEqual(e['expected_category_counts']['administration'],1)

    def test_expectations_do_not_influence_ranking(self):
        f=fixture();before=Matcher(f['catalogue']).rank(f['projects'][0]['text'])
        f['expectations'][0]['expected_top']='digital'
        self.assertEqual(before,Matcher(f['catalogue']).rank(f['projects'][0]['text']))


class PipelineTests(unittest.TestCase):
    def test_reproducible_outputs_and_sample_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a',Path(directory)/'b'
            r=run(ROOT/'data/sample/fixture.json',a);run(ROOT/'data/sample/fixture.json',b)
            self.assertEqual({p.name:p.read_bytes() for p in a.iterdir()},{p.name:p.read_bytes() for p in b.iterdir()})
            self.assertEqual(r['fixture_failures'],0)
            self.assertTrue(r['requires_human_review'])

    def test_bad_input_exit(self):self.assertEqual(main(['--input','missing.json']),1)

    def test_protect_input_directory(self):
        with self.assertRaises(ValueError):run(ROOT/'data/sample/fixture.json',ROOT/'data/sample')


if __name__=='__main__':unittest.main()
