import re
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

import warnings

dir_path = os.path.dirname(os.path.realpath(__file__))

# functions for parsing out the text from pipe-delimited files

# just read the file
def file_to_text(file_name):
    try:
        # default to UTF-8 and ignore non-utf8 char
        with open(os.path.join(dir_path, file_name), "r", encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception:
        print("Error with file {}".format(os.path.join(dir_path, file_name)))
        raise Exception

# this is where we normalize the text itself
def normalize(text):
    # remove line spaces, pipes and carrots
    normal = re.sub('[\|\n\r\^]', ' ', text)
    # remove \x0b, null, and nan
    normal = normal.replace("\x0b"," ").replace(" null ", " ").replace(" nan ", " ")
    # remove most punctuation (everything but period, dash, word and white space characters)
    normal = re.sub('[^\w\s\.\-\;\:\,]', ' ', normal)
    # remove long numbers
    normal = re.sub("\d{4,}", " ", normal)    
    # replace consecutive spaces with a single space
    normal = re.sub('\s{2,}', ' ', normal)
    return normal.lower().strip()

def extract_patient_data(line, date):
    pid_dict = {}
    pid_dict['pid'] = line[3]
    pid_dict['dob'] = re.sub("[\-\s\_]","",line[7])[:8]
    pid_dict['gender'] = line[8]
    if date and date != '':
        date = datetime.strptime(date, "%Y%m%d")
        dob = datetime.strptime(pid_dict['dob'], "%Y%m%d")
        pid_dict['age'] = round((date - dob).days / 365.0, 2)
    else:
        pid_dict['age'] = np.nan
    return pid_dict

def grab_text(text):
    split_text = text.split("|")
    good_text = []
    raw_text = []
    for field in split_text:
        if " " in field:
            good_text.append(field)
        raw_text.append(field)
    return " ".join(raw_text).strip(), " ".join(good_text).strip()

# rips the text out of the file, looks to assign labels, and returns a dictionary
def extract_text_hca( file, max_files=5000000000 ): 
    return hca_extractor(file, max_files=max_files)
    
# rips the text out of the file, looks to assign labels, and returns a dictionary
def extract_text(file_name): 
    file_data = data_extractor_simple(file_name)
    all_text = file_data['raw_text']
    # this finds those that have been positively labeled
    if "qb_y" in all_text.lower() or "[lngnod]" in all_text.lower() or "afnrzga19b" in all_text.lower():
        label = 1
    elif "qb_n" in all_text.lower():
        label = 0
    else:
        label = -1
    # make a unique id based on the file name
    _id = file_name.split("/")[-1].split(".txt")[0]
    file_data.update({
        'label': label,
        'id': _id
    })
    return file_data

def no_nodule_checker(text):
    no_nodules_regex = r"no pulmonary nodules|no pulmonary mass"
    no_nodules_regex += "|no chest radiographic evidence of focal airspace opacity or pulmonary nodules"
    no_nodules_regex += "|no concerning pulmonary nodule|no concerning pulmonary mass"
    no_nodules_regex += "|no worrisome pulmonary nodules|no large pulmonary nodules"
    no_nodules_regex += "|no discrete pulmonary nodules|no masses or discrete pulmonary nodules"
    no_nodules_regex += "|no suspicious pulmonary nodule|no nodules or masses"
    no_nodules_regex += "|no chest radiographic evidence of pulmonary nodules"
    no_nodules_regex += "|no evidence of lung nodule"
    if re.search(no_nodules_regex, text):
        return 1
    else:
        return 0
    
def data_extractor_simple(file_name, text="", hca=False):
    if text == "":
        raw_text = file_to_text(file_name)
    else:
        raw_text = text
    all_text = []
    lines = raw_text.split("\n")
    report_code = ""
    report_name = ""
    text = ""
    recommendations = False
    date = ""
    pid_dict = {}
    num_text_lines = len([line for line in lines if line.split("|")[0] == 'OBX'])
    for idx, line in enumerate(lines):
        if line == '\x1c':
            break
        split_line = line.split("|")
        if 'MSH' in split_line[0]:
            date = re.sub("[\-\s\_]","",split_line[6][:8])
        elif split_line[0] == 'OBR':
            report_code = split_line[4].split("^")[0].strip()
            report_name = re.sub('\s{2,}',' '," ".join(split_line[4].split("^")[1:])).strip()
        elif split_line[0] == 'OBX':
            if len(split_line) < 6:
                continue
            line_text = split_line[5].strip()
            if line_text == "":
                continue
            if "PDF" in line_text and "Base64" in line_text:
                break
            low_case = line_text.lower()
            if "~" in line_text and num_text_lines == 1:
                all_text.append(" ".join(line_text.split("~")))              
            elif "FLEISCHNER SOCIETY RECOMMENDATIONS:" not in line_text.upper():
                all_text.append(split_line[5])
        elif split_line[0] == 'PID':
            pid_dict = extract_patient_data(split_line, date)
        elif hca:
            for field in split_line:
                if " " in field:
                    all_text.append( field )
    text = " ".join(all_text)
    normalized_text = normalize(text)
    if "communicator weblaunch" in normalized_text:
        normalized_text = ""
    file_data = {
        'raw_text': raw_text.strip(),
        'text': normalized_text,
        'report_code': report_code,
        'report_name': report_name,
        'report_date': date,
        'no_nodules': no_nodule_checker(normalized_text)
    }
    file_data.update(pid_dict)
    return file_data

def data_extractor(file_name):
    raw_text = file_to_text(file_name)
    chest_text, all_text = [], []
    lines = raw_text.split("\n")
    report_code = ""
    report_name = ""
    text = ""
    chest = False
    chest_marker = 0
    recommendations = False
    date = ""
    pid_dict = {}
    num_text_lines = len([line for line in lines if line.split("|")[0] == 'OBX'])
    findings_indices = [idx for idx,line in enumerate(lines) if "findings:" in line.lower()]
    findings_index = findings_indices[0] if len(findings_indices) > 0 else None
    for idx, line in enumerate(lines):
        if line == '\x1c':
            break
        split_line = line.split("|")
        if 'MSH' in split_line[0]:
            date = re.sub("[\-\s\_]","",split_line[6][:8])
        elif split_line[0] == 'OBR':
            report_code = split_line[4].split("^")[0].strip()
            report_name = re.sub('\s{2,}',' '," ".join(split_line[4].split("^")[1:])).strip()
        elif split_line[0] == 'OBX':
            if (findings_index and idx < findings_index) or len(split_line) < 6:
                continue
            line_text = split_line[5].strip()
            if line_text == "":
                continue
            if "PDF" in line_text and "Base64" in line_text:
                break
            low_case = line_text.lower()
            lung_phrases = ['lungs', 'lungs and pleural spaces', 'lung bases', 'lungs and airways',
                            'lung parenchyma and pleura', 'lung parenchyma', 'lung pleura',
                           'lines tubes and devices', 'pulmonary nodules', 'lung rads', 'tubes and lines',
                           'lungs and pleural', 'lungs airways and pleural', 'lung and large airways']
            lung_checker = "|".join([x + ":" for x in lung_phrases])
            lung_checker_simple = "|".join(["lung","pulmonar","airway","tubes","lines","pleura","parenchyma"])
            stats_phrases = ['image number', 'average diameter', 'density', 'morphology', 'calcification',
                                   '# of nodules', 'nodule location', 'size of solid component']
            stats_checker = "|".join([x + ":" for x in stats_phrases])
            if re.search(lung_checker, low_case.replace(",","").replace(" (qb_y)","")):
                chest_marker = 1
                chest = True
                if "~" in line_text and num_text_lines == 1:
                    lung_index = None
                    next_up_index = None
                    marker_indices = []
                    for idx, text in enumerate(line_text.split("~")):
                        if text.upper().startswith("LUNGS:"):
                            lung_index = idx
                        elif ":" in text and lung_index:
                            next_up_index = idx
                            break
                    if next_up_index:
                        lung_text = " ".join(line_text.split("~")[lung_index:next_up_index])
                    else:
                        lung_text = " ".join(line_text.split("~")[lung_index:])
                    chest_text.append(lung_text)
                    continue                
            elif ":" in line_text[:min(35,len(line_text))] and not re.search(lung_checker_simple, low_case.split(":")[0]) \
                and not re.search(stats_checker, low_case):
                chest = False
            if "FLEISCHNER SOCIETY RECOMMENDATIONS:" in line_text.upper():
                recommendations = True
            if chest:
                chest_text.append(split_line[5])
            elif recommendations == False:
                all_text.append(split_line[5])
        elif split_line[0] == 'PID':
            pid_dict = extract_patient_data(split_line, date)
    if len(chest_text) == 0:
        text = " ".join(all_text)
    else:
        text = " ".join(chest_text)
    normalized_text = normalize(text)
    if "communicator weblaunch" in normalized_text:
        normalized_text = ""
    file_data = {
        'raw_text': raw_text.strip(),
        'text': normalized_text,
        'report_code': report_code,
        'report_name': report_name,
        'chest_marker': chest_marker,
        'report_date': date,
        'no_nodules': no_nodule_checker(normalized_text)
    }
    file_data.update(pid_dict)
    return file_data


# main function for reading in files, normalizing text, assign labels, and adding the records to a DataFrame
def files_to_df(directory, to_csv=True, max_files=10000):
    data = []
    for file in os.listdir(directory)[:max_files]:
        # just grab the text files that we want
        if file.endswith(".txt") and "-." not in file:
            file_dict = extract_text(directory + "/" + file)
            if file_dict != None and file_dict['text'] != "" and file_dict['text'] != None:
                data.append(file_dict)   
    df = pd.DataFrame(data).query("text == text")
    df['directory'] = directory.replace("/","_")
    if to_csv:
        out_to_csv(df, directory, file_type)
    return df

          
def out_to_csv(df, directory, file_type):
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    if file_type:
        file_name = directory.replace("/","_") + file_type + "_parsed_" + timestamp + ".csv"
    else:
        file_name = directory.replace("/","_") + "_parsed_" + timestamp + ".csv"
    df.to_csv(file_name)
    print("Finished parsing {} records, wrote to '{}'.".format(len(df), file_name))

# special extractor for files from HCA directory
def is_vetted_df(all_text):
    if "qb_y" in all_text.lower() or "[lngnod]" in all_text.lower() \
            or "afnrzga19b" in all_text.lower():
        label = 1
    elif "qb_n" in all_text.lower():
        label = 0
    else:
        label = -1
    return label


def hca_extractor(file, max_files=70000):    
    counter = 0
    if file.endswith(".txt") and "-." not in file and os.stat(file).st_size != 0:
        try:
            ## if lines are blank remove them
#             with open( file, "w" ) as f:
#                 for line in f:
#                     if not line.isspace():
#                         f.write(line)
            ## get first line to count how many columns there are
            with open( file ) as f:
                first_line = f.readline()
            if first_line.count("|") == 35 or first_line.count("|") == 34 :
                hca_h_use = hca_header
            elif first_line.count("|") == 5 or first_line.count("|") == 6 :
                return obx_extractor( file )
            else:
                print(file,first_line.count("|"), first_line)
            ##
            try:
                grouped_by_patient = pd.read_csv(file, names=hca_h_use, sep="|", dtype=np.object,
                                      usecols = hca_columns_wanted  )
            except Exception as e:
                grouped_by_patient = pd.read_csv(file, names=hca_header, sep="|", dtype=np.object,  )
                grouped_by_patient = grouped_by_patient[hca_columns_wanted]
            #### process data ###
            grouped_by_patient.fillna(value='', inplace=True)
            # PAT_ACCT_NUM
            grouped_by_patient = grouped_by_patient.groupby( by='PAT_ACCT_NUM'  ).agg({
#                                     'Record Number':'min',
                                    'dob': lambda x: " ".join([str(i) for i in set(x) ]),
                                    'sex': lambda x: " ".join([str(i) for i in set(x) ]),
                                    'Diagnosis Date': lambda x: " ".join([str(i) for i in set(x) ]),
                                    'Diagnosis Description': lambda x: " ".join([str(i) for i in set(x) ]),
                                    'procedure_text': lambda x: " ".join([str(i) for i in set(x) ]),
                                    'Diag Type': lambda x: " ".join([str(i) for i in set(x) ])
                                        }).reset_index()
            grouped_by_patient['raw_text'] = grouped_by_patient['Diagnosis Description'].astype(str) + grouped_by_patient['procedure_text'].astype(str)
            grouped_by_patient['text'] = grouped_by_patient['raw_text'].apply(lambda x: normalize(str(x) ) )
            grouped_by_patient['no_nodules'] = grouped_by_patient['text'].apply(lambda x: no_nodule_checker(x) )
            grouped_by_patient['label'] = grouped_by_patient['raw_text'].apply(lambda x: is_vetted_df(str(x) ) ) 
            grouped_by_patient['id'] = file.split("/")[-1].split(".txt")[0]+ grouped_by_patient['Record Number']
            grouped_by_patient['report_code'] = ""
            grouped_by_patient['report_name'] = ""
            grouped_by_patient.rename( columns= {"PAT_ACCT_NUM": "pid","sex":"gender",
                                                'Diagnosis Date': 'report_date'}, inplace=True  )
            return grouped_by_patient[['id','raw_text','text','report_code','report_name',
                           'report_date','pid','no_nodules']]                           
        except Exception as e:
            print("Error with file {}".format(file))
            raise Exception(e)
    return pd.DataFrame()


def obx_extractor(file ):    
    counter = 0
    if file.endswith(".txt") and "-." not in file:
        try:
            ##
            warnings.simplefilter("ignore")
            grouped_by_patient = pd.read_csv(file, names=obx_head, sep="|", dtype=np.object  )
            #### process data ###
            grouped_by_patient.fillna(value='', inplace=True)
            # PAT_ACCT_NUM
            grouped_by_patient = grouped_by_patient.groupby( by=['Record Number', 'Facility_Mnemonic_CS', 
                                                                 'MRN', 'URN', 'PAT_ACCT_NUM']  ).agg({
                             'Observation_text': lambda x: " ".join([str(i).replace("^"," ").replace(",","") for i in set(x) ]),
                                        }).reset_index()
            for c in obx_head:
                if c not in grouped_by_patient:
                    grouped_by_patient[c] = ''
            grouped_by_patient['Name'] = grouped_by_patient['Observation_text'].str.extractall(r'.+PATIENT NAME:\s+(.*)^')
            grouped_by_patient['raw_text'] = grouped_by_patient['Observation_text'].astype(str).replace("  "," ")
            grouped_by_patient['text'] = grouped_by_patient['raw_text'].apply(lambda x: normalize(str(x) ) )
            grouped_by_patient['no_nodules'] = grouped_by_patient['text'].apply(lambda x: no_nodule_checker(x) )
            grouped_by_patient['label'] = grouped_by_patient['raw_text'].apply(lambda x: is_vetted_df(str(x) ) ) 
            grouped_by_patient['id'] = file.split("/")[-1].split(".txt")[0]+ grouped_by_patient['Record Number']
            grouped_by_patient['report_name'] = ""
            grouped_by_patient['gender'] = ""
            grouped_by_patient['report_date'] = grouped_by_patient['Observation_text'].str.extract(r"EXAM DATE: (\d+/\d+/\d+)")
            grouped_by_patient.rename( columns= {"PAT_ACCT_NUM": "pid", "MRN":"report_code" }, inplace=True  )
            return grouped_by_patient[['id','raw_text','text','report_code','report_name', 'report_date','pid','no_nodules']]         
        except Exception as e:
            print("Error with file {}".format(file))
            raise Exception(e)
    return pd.DataFrame()

hca_columns_wanted = ['Record Number', 'dob', 'sex','PAT_ACCT_NUM', 'Diagnosis Date', 'Diagnosis Description',
                      'procedure_text', 'Diag Type']
groupby_hcas = ['Record Number','PAT_ACCT_NUM' ]

obx_head = ['Record Number', 'Facility_Mnemonic_CS', 'MRN', 'URN', 'PAT_ACCT_NUM', 'Observation_text']

hca_header= ['Record Number','Facility_Mnemonic_CS', 'MRN', 'URN', 'PAT_ACCT_NUM', 'first_name', 
             'middle_name', 'last_name', 'dob', 'sex', 'street_address', 'ADDR_LINE2_DESC', 'city', 
             'state', 'zip', 'country', 'Race', 'Ethenicty', 'Marital Status', 'Diag CPTCode', 
             'Diagnosis Date', 'Diagnosis Description', 'Diag Type', 'Insurance Code', 'Insurance_Group_Name', 
             'Height_in_Ft', 'Height_in_Inc', 'Height_Cap_Ft_Date', 'Weight_in_Lb', 'Weight_in_Oz', 'Weight_in_Kg', 
             'Weight_Cap_Date', 'Smoke Status', 'Some Status Capt Dt', 'procedure_text','Diag CPTCode2']