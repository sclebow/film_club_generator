import gzip
import os
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import main
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


FIXTURES = {
    'title.basics.tsv.gz': (
        'tconst\ttitleType\tprimaryTitle\tstartYear\tgenres\n'
        'tt1\tmovie\tFirst\t2000\tDrama\n'
        'tt2\tmovie\tSecond\t2001\tDrama\n'
        'tt3\tmovie\tFuture\t9999\tDrama\n'
        'tt4\tshort\tShort\t2001\tDrama\n'
        'tt5\tmovie\tUnknown year\t\\N\tDrama\n'
        'tt6\tmovie\tInvalid year\tnotayear\tDrama\n'
    ),
    'title.ratings.tsv.gz': (
        'tconst\taverageRating\tnumVotes\n'
        'tt1\t8.0\t100\n'
        'tt2\t6.0\t300\n'
        'tt3\t9.0\t999\n'
        'tt4\t9.0\t999\n'
        'tt5\t9.0\t999\n'
    ),
    'title.crew.tsv.gz': (
        'tconst\tdirectors\n'
        'tt1\tnm1\n'
        'tt2\tnm1\n'
    ),
    'title.principals.tsv.gz': (
        'tconst\tnconst\tcategory\n'
        'tt1\tnm2\tactor\n'
        'tt2\tnm2\tactress\n'
        'tt1\tnm3\tproducer\n'
        'tt1\tnm3\tproducer\n'
        'tt2\tnm3\tproducer\n'
        'tt1\tnm4\tproducer\n'
        'tt3\tnm4\tproducer\n'
        'tt4\tnm4\tproducer\n'
        'tt5\tnm4\tproducer\n'
        'tt6\tnm4\tproducer\n'
        'tt1\tnm5\twriter\n'
    ),
    'name.basics.tsv.gz': (
        'nconst\tprimaryName\tbirthYear\tdeathYear\tprimaryProfession\n'
        'nm1\tDirector\t1970\t\\N\tdirector\n'
        'nm2\tActor\t1980\t\\N\tactor\n'
        'nm3\tProducer\t1960\t\\N\tproducer\n'
        'nm4\tOther Producer\t1965\t\\N\tproducer\n'
        'nm5\tWriter\t1975\t\\N\twriter\n'
    ),
}


class ProducerAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        st.cache_data.clear()
        for name, contents in FIXTURES.items():
            with gzip.open(name, 'wt', encoding='utf-8') as file:
                file.write(contents)

    def tearDown(self):
        st.cache_data.clear()
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_build_counts_distinct_released_producer_movies_and_retains_existing_roles(self):
        main.create_filtered_dataset()
        producers, movies, stats = main.find_producers_with_n_movies(2)
        self.assertEqual(producers['primaryName'].tolist(), ['Producer'])
        self.assertEqual(set(movies['tconst']), {'tt1', 'tt2'})
        self.assertEqual(len(movies), 2)
        self.assertEqual(stats.set_index('producer_id').loc['nm3', 'total_votes'], 400)
        self.assertEqual(stats.set_index('producer_id').loc['nm3', 'avg_rating'], 7)
        self.assertEqual(stats.set_index('producer_id').loc['nm4', 'movie_count'], 1)
        self.assertEqual(main.find_directors_with_n_movies(2)[0]['primaryName'].tolist(), ['Director'])
        self.assertEqual(main.find_actors_with_n_movies(2)[0]['primaryName'].tolist(), ['Actor'])
        self.assertTrue(main.find_producers_with_n_movies(3)[0].empty)

    def test_old_cache_is_rebuilt_and_new_cache_is_reused(self):
        with open(main.FILTERED_DATA_FILE, 'wb') as file:
            pickle.dump({'actor_stats': 'old'}, file)
        rebuilt = main.load_and_process_base_data()
        self.assertEqual(len(rebuilt), 9)
        with open(main.FILTERED_DATA_FILE, 'rb') as file:
            self.assertIn('producer_stats', pickle.load(file))
        main.load_and_process_base_data.clear()
        with patch.object(main, 'create_filtered_dataset', side_effect=AssertionError('unexpected rebuild')):
            self.assertEqual(len(main.load_and_process_base_data()), 9)

    def test_build_filters_unrelated_rows_across_chunks(self):
        loader = main.download_and_load_imdb_data
        loaded = {}

        def capture(url, filename, **kwargs):
            result = loader(url, filename, chunksize=2, **kwargs)
            loaded[filename] = result.copy()
            return result

        with patch.object(main, 'download_and_load_imdb_data', side_effect=capture):
            main.create_filtered_dataset(parallel_load=False)
        self.assertEqual(loaded['title.basics.tsv.gz']['tconst'].tolist(), ['tt1', 'tt2'])
        self.assertEqual(loaded['title.ratings.tsv.gz']['tconst'].tolist(), ['tt1', 'tt2'])
        self.assertEqual(len(loaded['title.principals.tsv.gz']), 6)
        self.assertEqual(set(loaded['name.basics.tsv.gz']['nconst']), {'nm1', 'nm2', 'nm3', 'nm4'})

    def test_popularity_score_caps_vote_weight(self):
        contents = FIXTURES['title.ratings.tsv.gz'].replace('tt2\t6.0\t300', 'tt2\t6.0\t200000')
        with gzip.open('title.ratings.tsv.gz', 'wt', encoding='utf-8') as file:
            file.write(contents)
        result = main.create_filtered_dataset()
        for stats, id_column, person in [(result[1], 'director_id', 'nm1'),
                                         (result[4], 'actor_id', 'nm2'),
                                         (result[7], 'producer_id', 'nm3')]:
            self.assertEqual(stats.set_index(id_column).loc[person, 'popularity_score'], 700)

    def test_parallel_build_matches_sequential_and_runs_both_readers(self):
        sequential = main.create_filtered_dataset(parallel_load=False)
        barrier = threading.Barrier(2, timeout=10)
        reader = main.read_imdb_file
        worker_threads = set()

        def synchronized_reader(filepath, *args, **kwargs):
            if filepath.name in ('title.crew.tsv.gz', 'title.principals.tsv.gz'):
                worker_threads.add(threading.get_ident())
                barrier.wait()
            return reader(filepath, *args, **kwargs)

        with patch.object(main, 'read_imdb_file', side_effect=synchronized_reader):
            parallel = main.create_filtered_dataset()
        self.assertEqual(len(worker_threads), 2)
        for expected, actual in zip(sequential, parallel):
            pd.testing.assert_frame_equal(expected, actual)

    def test_parallel_worker_error_is_propagated(self):
        reader = main.read_imdb_file

        def failing_reader(filepath, *args, **kwargs):
            if filepath.name == 'title.crew.tsv.gz':
                raise ValueError('bad crew data')
            return reader(filepath, *args, **kwargs)

        with patch.object(main, 'read_imdb_file', side_effect=failing_reader):
            with self.assertRaisesRegex(ValueError, 'bad crew data'):
                main.create_filtered_dataset(parallel_load=True)
        self.assertFalse(Path(main.FILTERED_DATA_FILE).exists())

    def test_parallel_environment_setting_applies_only_to_builds(self):
        with patch.dict(os.environ, {'FILM_CLUB_PARALLEL_LOAD': '1'}):
            with patch.object(main, 'create_filtered_dataset', wraps=main.create_filtered_dataset) as build:
                main.load_and_process_base_data()
                build.assert_called_once_with(parallel_load=True)
            main.load_and_process_base_data.clear()
            with patch.object(main, 'create_filtered_dataset', side_effect=AssertionError('unexpected rebuild')):
                self.assertEqual(len(main.load_and_process_base_data()), 9)

    def test_build_defaults_to_parallel_and_can_be_disabled(self):
        for value, expected in [(None, True), ('1', True), ('0', False)]:
            for old_cache in (False, True):
                with self.subTest(value=value, old_cache=old_cache):
                    st.cache_data.clear()
                    filepath = Path(main.FILTERED_DATA_FILE)
                    if old_cache:
                        with filepath.open('wb') as file:
                            pickle.dump({'actor_stats': 'old'}, file)
                    else:
                        filepath.unlink(missing_ok=True)
                    with patch.dict(os.environ):
                        os.environ.pop('FILM_CLUB_PARALLEL_LOAD', None)
                        if value is not None:
                            os.environ['FILM_CLUB_PARALLEL_LOAD'] = value
                        with patch.object(main, 'create_filtered_dataset', return_value=()) as build:
                            main.load_and_process_base_data()
                            build.assert_called_once_with(parallel_load=expected)

    def test_no_producer_credits(self):
        contents = FIXTURES['title.principals.tsv.gz']
        with gzip.open('title.principals.tsv.gz', 'wt', encoding='utf-8') as file:
            file.writelines(line for line in contents.splitlines(keepends=True) if '\tproducer\n' not in line)
        main.create_filtered_dataset()
        producers, movies, stats = main.find_producers_with_n_movies(1)
        self.assertTrue(producers.empty)
        self.assertTrue(movies.empty)
        self.assertTrue(stats.empty)
        app = AppTest.from_file(str(Path(main.__file__).resolve()), default_timeout=30).run()
        app.sidebar.radio[0].set_value('Producers').run()
        self.assertFalse(app.exception)

    def test_producer_and_combined_ui(self):
        main.create_filtered_dataset()
        app = AppTest.from_file(str(Path(main.__file__).resolve()), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio[0].value, 'Both')
        app.sidebar.slider[0].set_value((2, 2))
        app.sidebar.radio[0].set_value('Producers').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('1 producers' in item.value for item in app.success))
        self.assertTrue(any('Producers List' in tab.label for tab in app.tabs))
        self.assertIn('Producer', app.dataframe[0].value['Name'].tolist())
        app.sidebar.radio[0].set_value('All').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('1 directors, 1 actors and 1 producers' in item.value for item in app.success))
        self.assertTrue(any('Producers' in tab.label for tab in app.tabs))

    def test_inclusive_movie_ranges_for_all_roles(self):
        basics = FIXTURES['title.basics.tsv.gz'] + 'tt7\tmovie\tThird\t2002\tDrama\n'
        crew = FIXTURES['title.crew.tsv.gz'] + 'tt1\tnm6\n'
        principals = FIXTURES['title.principals.tsv.gz'] + 'tt1\tnm7\tactor\ntt7\tnm2\tactor\ntt7\tnm3\tproducer\n'
        names = FIXTURES['name.basics.tsv.gz'] + (
            'nm6\tShort Director\t1970\t\\N\tdirector\n'
            'nm7\tShort Actor\t1970\t\\N\tactor\n'
        )
        for filename, contents in [('title.basics.tsv.gz', basics),
                                   ('title.crew.tsv.gz', crew),
                                   ('title.principals.tsv.gz', principals),
                                   ('name.basics.tsv.gz', names)]:
            with gzip.open(filename, 'wt', encoding='utf-8') as file:
                file.write(contents)
        main.create_filtered_dataset()
        for find, column, expected in [
            (main.find_directors_with_n_movies, 'director_id', {'nm1', 'nm6'}),
            (main.find_actors_with_n_movies, 'actor_id', {'nm2', 'nm7'}),
            (main.find_producers_with_n_movies, 'producer_id', {'nm3', 'nm4'}),
        ]:
            with self.subTest(role=column):
                people, movies, _ = find(1, 3)
                self.assertEqual(set(people[column]), expected)
                self.assertEqual(set(movies[column]), expected)
                self.assertTrue(set(people['movie_count']).issubset({1, 2, 3}))
                self.assertTrue(find(4, 5)[0].empty)
        self.assertEqual(main.find_actors_with_n_movies(3)[0]['primaryName'].tolist(), ['Actor'])
        self.assertEqual(main.find_actors_with_n_movies(3, 3)[0]['primaryName'].tolist(), ['Actor'])

    def test_range_slider_updates_all_analysis_views(self):
        main.create_filtered_dataset()
        app = AppTest.from_file(str(Path(main.__file__).resolve()), default_timeout=30).run()
        self.assertEqual(app.sidebar.slider[0].value, (12, 12))
        app.sidebar.slider[0].set_value((1, 2))
        for role, expected in [('Directors', '1 directors'), ('Actors', '1 actors'),
                               ('Producers', '2 producers'), ('Both', '1 directors and 1 actors'),
                               ('All', '1 directors, 1 actors and 2 producers')]:
            with self.subTest(role=role):
                app.sidebar.radio[0].set_value(role).run()
                self.assertFalse(app.exception)
                self.assertTrue(any(expected in item.value and '1–2 movies' in item.value
                                    for item in app.success))
                self.assertTrue(any('1–2 Movies' in item.label for item in app.metric))
                if role in ('Producers', 'All'):
                    producer_table = next(item.value for item in app.dataframe
                                          if 'Movie Count' in item.value.columns
                                          and set(item.value['Name']) == {'Producer', 'Other Producer'})
                    self.assertEqual(set(producer_table['Movie Count']), {1, 2})

    def test_range_highlight_marks_both_endpoints(self):
        from plotly.graph_objects import Figure

        figure = Figure()
        main.highlight_movie_range(figure, (10, 13))
        self.assertEqual(figure.layout.shapes[0].x0, 9.5)
        self.assertEqual(figure.layout.shapes[0].x1, 13.5)
        figure = Figure()
        main.highlight_movie_range(figure, (12, 12))
        self.assertEqual(figure.layout.shapes[0].x0, 12)
        self.assertEqual(figure.layout.shapes[0].x1, 12)


if __name__ == '__main__':
    unittest.main()
