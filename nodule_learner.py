import numpy as np
import pandas as pd
import re
import uuid
# import nltk
# nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import json
import warnings

import nodule_parser as parser
import psutil

import sys
import os

# If the code is frozen, use this path:
if getattr(sys, 'frozen', False):
    CurrentPath = sys._MEIPASS
# If it's not use the path we're on now
else:
    CurrentPath = os.path.dirname(__file__)

stemmer = SnowballStemmer("english")

from nodule_extractor import NoduleExtractor
ne = NoduleExtractor()

import sqlite3

class NoduleLearner(object):
    
    # initialize our Learner object
    # also loads the mapper object and the model itself, and creates the prefilter and stop words lists needed
    def __init__(self, timestamp='20180613_133127', cache_size=500, thread_lock=False):
        self.prefilter()
        self.stop_words_stemmed = self.get_stop_words()
        mapper_file = open("mapper_{}.pkl".format(timestamp), "rb")
        model_file = open("model_{}.pkl".format(timestamp), "rb")
        scaler_file = open("scaler_{}.pkl".format(timestamp), "rb")
        self.mapper = pickle.load(mapper_file)
        self.transformed_names = [term.replace("stemmed_","") for term in self.mapper.transformed_names_]
        self.clf = pickle.load(model_file)
        self.scaler = pickle.load(scaler_file)
        self.cache_size = cache_size
        self.counter = 0
        self.features_df = pd.DataFrame()
        self.thread_lock = thread_lock
        self.patient_data_sql_table = "patient_data"
        self.initialize_db()

        
    def initialize_db(self):
        if not isinstance(self.thread_lock, bool):
            # we assume that if thread_lock is set it is a multiprocessing.lock Or threading.Lock obj
            with self.thread_lock: 
                table_suffix = uuid.uuid4().hex
                self.patient_data_sql_table = self.patient_data_sql_table+str(table_suffix)
                self.conn = sqlite3.Connection("patient_data.sqlite")
                self.cur = self.conn.cursor()
                self.pd_fields = ['directory', 'filename', 'pid', 'max_nodule_size', 'report_date']
                self.cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
                tables = self.cur.fetchall()
                if self.patient_data_sql_table in tables:
                    self.patient_data = pd.read_sql("SELECT * FROM '%s'" % self.patient_data_sql_table, conn)
                else:
                    self.patient_data = pd.DataFrame({field:[] for field in self.pd_fields})
        else:
            self.conn = sqlite3.Connection("patient_data.sqlite")
            self.cur = self.conn.cursor()
            self.pd_fields = ['directory', 'filename', 'pid', 'max_nodule_size', 'report_date']
            self.cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
            tables = self.cur.fetchall()
            if self.patient_data_sql_table in tables:
                self.patient_data = pd.read_sql("SELECT * FROM '%s'" % self.patient_data_sql_table, conn)
            else:
                self.patient_data = pd.DataFrame({field:[] for field in self.pd_fields})
            
                
        
    # grabbing the stemmed stop words
    def get_stop_words(self):
        tfidf = TfidfVectorizer(stop_words='english')

        stop_words = list(tfidf.get_stop_words())

        stop_words += ['qb_y', 'qb_n', 'lngnod']

        stop_words_stemmed = [stemmer.stem(x) for x in stop_words]
        good_words = ['no', 'none', 'not', 'nothing', 'back', 'between', 'found', 'front', 'find',
                      'without', 'above', 'almost', 'under', 'among', 'together', 'serious', 'less',
                      'each', 'bottom', 'full', 'empty']
        good_words_stemmed = [stemmer.stem(x) for x in good_words]
        for word in good_words_stemmed:
            stop_words_stemmed.remove(word)
            
        return stop_words_stemmed
        
        
    # load the prefilter to run on incoming files
    # this ensures we have the right kinds of reports, i.e. those that actually talk about the chest/lungs
    def prefilter(self):
        # remove the text from those that have no indication of being about the chest or lungs
        prefilter_terms = ["71260", "71250", "71270", "71275", "75574", "78815", "chest", "lung",
                          "pulmonary", "hilum", "hilar", "mediastinum", "costal", "parenchyma", "pleura",
                          "fleischner", "pleural", "groundglass"]
        self.prefilter_string = "|".join(prefilter_terms)
        self.prefilter_regex = re.compile(self.prefilter_string)
        bad_report_types = ["parathyroid scan", "ct soft tissue neck"]
        self.bad_reports_regex = "|".join(bad_report_types)
        
    
    # run the prefilter regex; doesn't actually remove the files (we still want the model to make a prediction)
    # just removes the text itself, ensuring a negative prediction
    def prefilter_text(self, raw_text, text):
        if re.search(self.bad_reports_regex, raw_text.lower()) or not re.search(self.prefilter_regex, raw_text.lower()):
            return ""
        else:
            return text
        
    # transform text to prepare it for fitting to the model or for predicting
    def transform_text(self, rec, file, vetting=False):
        # extract the directory where the file came from, to disambiguate in case of duplicate file id's
        directory = "_".join(file.split("/")[:-1])
        # run the text through the prefilters
        prefiltered = self.prefilter_text(rec['raw_text'], rec['text'])
        # extract size, location, etc.
        extracted = ne.extract(prefiltered)
        # grab the data for the largest nodule, if one is found
        max_nodule_sizes = [x['max_nodule_size_numeric'] for x in extracted if x['max_nodule_size_numeric'] != '']
        max_nodule_size = 0
        max_nodule_location = ""
        max_nodule_lung = ""
        if max_nodule_sizes != []:
            nodule_size_present = 1
            max_nodule_size = max_nodule_sizes[0]
            max_nodule_location = extracted[0]['max_nodule_location']
            max_nodule_lung = extracted[0]['max_nodule_lung']
        else:
            nodule_size_present = 0            
        # grab the actual phrases where terms for nodule were found
        phrases = [x['nodule_phrase'] for x in extracted]
        # add nodule data to our output record
        rec.update({'nodule_phrases': " ".join(phrases), 'nodule_count': len(phrases),
                       'nodule_size_present': nodule_size_present})
        # stem the nodule phrases and report name
        rec['stemmed'] = self.stem(rec['text'])
        rec['stemmed_report_name'] = self.stem(rec['report_name'])
        # add in the data for the largest nodule, and also the filename and directory
        rec.update({'max_nodule_size': max_nodule_size, 'max_nodule_location': max_nodule_location,
                    'max_nodule_lung': max_nodule_lung, 'filename': rec['id'],
                   'directory': directory, 'max_nodule_change': 0, 'prev_max_date': ''})

        # add the filter_pass feature to indicate whether the record passed the filters
        if prefiltered == "":
            rec['filter_pass'] = 0
        else:
            rec['filter_pass'] = 1
            
        rec['truth_marking'] = rec.get('label',-1)
        if rec['truth_marking'] != np.nan:
            rec['truth_marking'] = int(rec['truth_marking']) # force value to int (instead of float)
            
        # if vetting, return the data for each individual nodule phrase found
        output = []
        if vetting:
            for counter, nodule in enumerate(extracted):
                temp_rec = {k:v for k,v in rec.items()}
                temp_nodule = {k:v for k,v in nodule.items() if not 'max' in k}
                temp_rec['text'] = rec.get('text','') ## added in to pass the text out for vetting reasons
                temp_rec.update(temp_nodule)
                # add a phrase counter to differentiate among them
                temp_rec['phrase_counter'] = counter + 1
                output.append(temp_rec)
            if output != []:
                return output
            else:
                rec.update({'nodule_size': '', 'nodule_location': '',
                            'nodule_size_numeric': 0.0, 'nodule_lung': '',
                            'is_nodule': 0, 'phrase_counter': 0, 'nodule_phrase': ''})
        return [rec]
        
    # transform files or text
    def transform_hca(self, file, vetting):
        parsed = parser.extract_text_hca(file, max_files=5000000000 )
        #
        if len(parsed) > 0:
            parsed['nodule_count'] = 0
            parsed['filename'] = file
            ### remove duplicates but keep 'seed rows' so we can expand back
            columns = list( parsed.columns )
            parsed['index_col'] = parsed.index
            if 'index_col' in columns:
                columns.remove('index_col')
            parsed = parsed.groupby(columns).agg({ 'index_col':lambda x: tuple(set(x)) }).reset_index()
            #
            ####
            parsed['prefiltered']=parsed.apply(lambda x: \
                                        self.prefilter_text(x['raw_text'], x['text']), axis=1 )
            #parsed['prefiltered'] = parsed['text'] # if prefilter is messing us up
            # extract size, location, etc.
            parsed['extracted'] = parsed['prefiltered'].apply( lambda x: ne.extract(x) )
            #
            # grab the data for the largest nodule, if one is found
            parsed['max_nodule_sizes'] =  parsed['extracted'].apply(lambda i: [x['max_nodule_size_numeric'] for x in i if x['max_nodule_size_numeric'] != '']  )
            parsed['max_nodule_size'] = 0
            parsed['max_nodule_location'] = ""
            parsed['max_nodule_lung'] = ""
            parsed['max_nodule_change'] = 0
            parsed['prev_max_date'] = ''
            #
            parsed['nodule_size_present'], parsed['max_nodule_size'], parsed['max_nodule_location'], parsed['max_nodule_lung']= \
                        zip(*parsed.apply(lambda x: self.parse_node_sizes( x ), axis=1 ))
            # grab the actual phrases where terms for nodule were found
            parsed['phrases'] = parsed['extracted'].apply(lambda i: [x['nodule_phrase'] for x in i] )
            parsed['nodule_phrases'] = parsed['phrases'].apply(lambda i: " ".join(i) )
            parsed['nodule_count'] = parsed['phrases'].apply(lambda i: len(i) )
            # stem the nodule phrases and report name
            parsed['stemmed'] = self.stem_vector( parsed[~pd.isnull(parsed.text)].text  )
            parsed['stemmed_report_name'] = self.stem_vector( parsed[~pd.isnull(parsed.report_name)].report_name )
            # add the filter_pass feature to indicate whether the record passed the filters
            parsed['filter_pass'] = 0
            parsed.loc[parsed['prefiltered']!= "",'filter_pass'] = 1
            #
            #print("debuggerrrr2 ",   parsed.iloc[0] )  
            if 'label' in parsed:
                parsed['truth_marking'] = parsed['label']
            else:
                parsed['truth_marking'] = -1
            #####
            #  not dealing with Vetting ... merrp ##
            #####
            #parsed['no_nodules'] = 0
            #self.features_df = self.features_df.append( parsed )
            self.features_df = parsed
        
    # transform files or text
    def transform(self, file, vetting):
        parsed = parser.extract_text(file)
        transformed = self.transform_text(parsed, file, vetting)
        self.features_df = self.features_df.append(pd.DataFrame(transformed))
        
    # make predictions with optional probability output
    def predict(self, probability=False, vetting=False):
        ''' transfrom and predict from our features...
            this is the real time suck of the entire process'''
        def chunker(seq, size):
            ''' this is used to chunk through the features_df 
            we use a generator to be as light as possible.
            ''' 
            for pos in range(0, len(seq), size):
                yield seq[pos:pos + size]      #.copy()
        output_list = [] # this var holds all the resulting rows for output
        
        for df_chunk in chunker(self.features_df, 200000):
            mapped_features = self.mapper.transform(df_chunk) # this line is grossly memmory inefficient -can't fix though :/ hence we stream chunks through instead
            if probability:
                predictions = self.clf.predict_proba(mapped_features)
            else:
                predictions = self.clf.predict( mapped_features )
            #print("DEBUGGER predictions", predictions)
            mapped_features = None # clear for space
            # set the columns to include in the output
            output_cols = ['filename', 'directory', 'nodule_phrases', 'max_nodule_size',
                           'max_nodule_location', 'max_nodule_lung', 'max_nodule_change',
                           'prev_max_date', 'report_date', 'pid', 'truth_marking', 'index_col']
            # set columns only used when vetting the data
            vetting_cols = ['nodule_size', 'nodule_location', 'nodule_size_numeric',
                    'nodule_lung', 'is_nodule', 'nodule_phrase', 'phrase_counter', 'text']
            if vetting:
                output_cols += vetting_cols
            #
            for c in output_cols:
                if c not in df_chunk.columns:
                    df_chunk.loc[:,c] = np.NaN
            output_df = df_chunk.loc[:, output_cols]
            # set the 'evidence' value equal to the 'nodule_phrases'
            output_df['evidence'] = output_df['nodule_phrases']
            output_df = output_df.drop('nodule_phrases', axis=1)
            # grab the probability and prediction value if probability is selected
            # otherwise just output the prediction value (0 or 1)
            if probability:
                output_df['probability'] = [round(max(pred),3) for pred in predictions]
                output_df['prediction'] = [int(np.argmax(pred)) for pred in predictions]
            else:
                output_df['prediction'] = predictions
                output_df['prediction'] = output_df['prediction'].astype(int)
            # for values where the prediction is 1 (positive), check the growth rate
            # otherwise reset the nodule extraction data
            # expand rows from index_col
            if list(df_chunk.index_col.values)[0] != np.NaN:
                output_df = self.split_data_frame_list( output_df, 'index_col' )
            output_df.drop('index_col', axis=1, inplace=True) # now lets remove the grouper index columns
            ##
            for idx, row in output_df.iterrows():
                temp_dict = dict(row)
                if row['prediction'] == 1:
                    if row['max_nodule_size'] > 0:
                        temp_dict = self.check_growth(temp_dict)
                else:
                    for field in ['max_nodule_lung', 'max_nodule_location']:
                        temp_dict[field] = ""
                    temp_dict['max_nodule_size'] = 0
                    #temp_dict = self.check_growth(temp_dict) #remove this line and uncoment above FIXME deubgMode
                output_list.append(temp_dict)     
        # END CHUNK_LOOP reset the features_df and mapped_features dataframes
        self.features_df = pd.DataFrame()
        self.mapped_features = pd.DataFrame()
        # output the results as a JSON array
        return json.dumps(output_list)
        
    # transform input data and make predictions in a single function
    def transform_predict(self, file, probability=False, vetting=False, hca=False):
        # ensure file can be read into memory if not dump & reset counter
        if psutil.virtual_memory().available - (os.path.getsize(file) * 2 ) < (1024*10):
            return self.dump_predictions(probability, vetting)
        #
        if hca:
            self.transform_hca(file=file, vetting=vetting)
            return self.dump_predictions(probability, vetting)
        else:
            self.transform(file=file, vetting=vetting)
        # use counter to manager cache
        # if number is larger than cache_size, make the predictions
        # otherwise just increment the counter and return an empty JSON array
        if self.counter >= self.cache_size:
            return self.dump_predictions(probability, vetting)
        elif psutil.virtual_memory().available - (os.path.getsize(file) * 2 ) < (1024*10):
            return self.dump_predictions(probability, vetting)
        else:
            self.counter += 1
            return json.dumps([])
        
    # check for increase/decrease of size of largest nodule
    def check_growth(self, rec):
        self.patient_data = self.patient_data.append(pd.DataFrame(
            {field:[rec[field]] for field in self.pd_fields}))

        if not self.patient_data.query("pid == '{}'".format(rec['pid'])).empty:
            prev_history = self.patient_data.query(
                "pid == '{}'".format(rec['pid'])).sort_values('report_date', ascending=False)
            prev_max_nodule = prev_history['max_nodule_size'].values[0]
            rec['max_nodule_change'] = rec['max_nodule_size'] - prev_max_nodule
            rec['prev_max_date'] = prev_history['report_date'].values[0]
            
        return rec
        
    # dump predictions out for all the cached records
    def dump_predictions(self, probability=False, vetting=False ):
        # write current patient data to SQL table
        if not isinstance(self.thread_lock, bool):
            # we assume that if thread_lock is set it is a multiprocessing.lock Or threading.Lock obj
            with self.thread_lock: 
                self.patient_data.to_sql(self.patient_data_sql_table, self.conn, if_exists='replace')
        else:
            self.patient_data.to_sql(self.patient_data_sql_table, self.conn, if_exists='replace')
        # if there's nothing left to predict on, just output an empty JSON array
        if self.features_df.empty:            
            return json.dumps([])
        # scale the nodule count using the scaler
        self.features_df['nodule_count_scaled'] = self.scaler.transform(self.features_df[['nodule_count']])
        # reset the counter
        self.counter = 0
        # modified to batch through data in predict function
        warnings.simplefilter("ignore")
        return self.predict(probability=probability, vetting=vetting)      
        
    # normalize text
    def normalize(self, text):
        text = text.lower()
        text = re.sub("qb_y|qb_n", "", text)
        text = re.sub('[\:\;\.\,\-\_]', ' ', text)
        text = re.sub('\d', ' ', text)
        text = re.sub('\s{2,}', ' ', text)
        return text

    # stem text
    def stem(self, text):
        normalized = self.normalize(text)
        tokenized = word_tokenize(normalized)
        stemmed = [stemmer.stem(word) for word in tokenized]
        cleaned = [word for word in stemmed if word not in self.stop_words_stemmed]
        return " ".join(cleaned)
    
    # stem vectors of string
    def stem_vector(self, text):
        normalized = text.str.lower().replace( {'qb_y|qb_n': '', '[\:\;\.\,\-\_]': '', '\d': ' ', '\s{2,}': ' '},regex=True )
        tokenized = normalized.apply(lambda x: word_tokenize(x))
        stemmed = tokenized.apply(lambda t: [stemmer.stem(word) for word in t] )
        cleaned = [" ".join(word) for word in stemmed if word not in self.stop_words_stemmed and word !=np.nan and word != ""]
        return cleaned
    
    def split_data_frame_list(self, df, target_column):
        ''' 
        df: dataframe to split
        target_column: the column containing the values to split
        returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
        The values in the other columns are duplicated across the newly divided rows.
        '''
        row_accumulator = []
        def split_list_to_rows(row):
            split_row = row[target_column]
            if isinstance(split_row, tuple) or  isinstance(split_row, list):
                for s in split_row:
                    new_row = row.to_dict()
                    new_row[target_column] = s
                    row_accumulator.append(new_row)
            else:
                new_row = row.to_dict()
                new_row[target_column] = split_row
                row_accumulator.append(new_row)
        df.apply(split_list_to_rows, axis=1)
        new_df = pd.DataFrame(row_accumulator)
        return new_df
    
    def parse_node_sizes(self, row):
        max_nodule_sizes = row.get("max_nodule_sizes",[])
        extracted= row.get("extracted",[{"max_nodule_location":"", "max_nodule_lung":""}])
        #print( "DEBUGGING:%s extracted= " % (row['pid']), json.dumps(extracted) )
        #print()
        nodule_size_present = 0
        max_nodule_size = 0
        max_nodule_location=''
        max_nodule_lung = ''
        if len(extracted)>0 and extracted[0].get('max_nodule_location','') != '':
            #print("DEBUGGER_parse2", row.get('pid',''),extracted[0]['max_nodule_location'] , extracted[0]['max_nodule_lung']    ) 
            fflag=False
        if max_nodule_sizes != []:
            nodule_size_present = 1 # len(list(max_nodule_sizes[0].unique() ))
            max_nodule_size = max_nodule_sizes[0]
            max_nodule_location = extracted[0]['max_nodule_location']
            max_nodule_lung = extracted[0]['max_nodule_lung']  
            ## assume if a previous exists with same measurement and location it is the same
            if max_nodule_location == "":
                #print( "DEBUGGING:%s extracted= " % (row['pid']), json.dumps(extracted) )
                nod_index=0
                for extracted_cnt, extacted_stuff in enumerate(extracted):
                    if extacted_stuff['nodule_size_numeric'] == max_nodule_size and \
                        extacted_stuff['nodule_location'] != "" and nod_index ==0:
                        nod_index = extracted_cnt
                max_nodule_location = extracted[nod_index]['nodule_location']
                max_nodule_lung = extracted[nod_index]['nodule_lung']   
        return nodule_size_present, max_nodule_size, max_nodule_location, max_nodule_lung

if __name__ == "__main__":    
    if (sys.argv.__len__() > 1):
        parsingFileName = sys.argv[1].replace("/", " ")
        nl = NoduleLearner()
        prediction = nl.transform_predict(os.path.join(CurrentPath, parsingFileName), False, True)
        print(prediction)



