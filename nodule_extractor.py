import re
from operator import itemgetter

# class for extracting size, location, lung of identified nodules
class NoduleExtractor(object):
    
    
    def __init__(self):
        # compiles regex objects to use for extracting
        self.initialize_regex()

    def extract(self, text, keep_largest=False):
        phrases = self.get_nodule_sentences(text)
        extracted_nodules = self.extract_nodules(phrases)
        if keep_largest:
            sorted_nodules = sorted(extracted_nodules, key=itemgetter('nodule_size_numeric'), reverse=True)
            return sorted_nodules[0]
        return extracted_nodules
    
    # splits the text on sentences
    def get_sentences(self, text):
        sentences = [sentence.strip() for sentence in re.split('(\.[\sA-Za-z])', text)]
        return sentences
    
    # grabs all the sentences that contain a predetermined term for 'nodule'
    def get_nodule_sentences(self, text):
        
        sentences = self.get_sentences(text)
        
        # nodule checker
        nodule_terms = ["nodules", "nodule", "nodular", "groundglass", "opacification", "spiculated",
                        "granulomas", "granuloma", "fdg", "neoplasm", "masses", "mass", "lesion" ] 
        # Added for OBX - lesion
        nodule_checker = re.compile("(" + "|".join(nodule_terms) + ")")
        
        nodule_sentences = []
        for sentence in sentences:
            if nodule_checker.search(sentence):
                if len(nodule_checker.findall(sentence)) > 1 and re.search("[\,\;]", sentence):
                    sentences += re.split("[\,\;]", sentence)
                else:
                    nodule_sentences.append(sentence.strip())
        
        # this will try to make sure the nodule refers to a lung nodule
        lung_terms = ['lungs', 'lung', 'pulmonary']
        lung_checker = re.compile("|".join(lung_terms))
        
        lung_nodule_sentences = [{'nodule_phrase': sen} for sen in nodule_sentences]# if lung_checker.search(sen)]
        
        return lung_nodule_sentences
        
    def initialize_regex(self):
        # create regex to identify locations in the lungs
        self.locations = ["left upper lobe", "lingula", "left lower lobe", "right upper lobe",
                     "right middle lobe", "right lower lobe", "left lobe base", "right lobe base",
                          "left lower basilar", "lung base", # OBX
                         "left lobe apex", "right lobe apex", 'rul', 'rml', 'rll', 'lul', 'lll']
        self.locations_dict = {'left lobe base': 'left lower lobe', 'left lung base': 'left lower lobe',
                               'left lower basilar' : 'left lower lobe', "lung base":'lower lobe', #OBX 
                               'right lobe base': 'right lower lobe', 'right lung base': 'right lower lobe',
                               'left lobe apex': 'left upper lobe', 'left lung apex': 'left upper lung',
                               'right lobe apex': 'right upper lobe', 'right lung apex': 'right upper lobe',
                              'rul': 'right upper lobe', 'rml': 'right middle lobe', 'rll': 'right lower lobe',
                              'lul': 'left upper lobe', 'lll': 'left lower lobe'}
        nodule_locs_re = []
        for loc in self.locations:
            if len(loc) == 3:
                loc = "(\s|^)" + loc + "(\s|$|\.|\,|\;)"
            nodule_locs_re.append(re.compile("(" + loc.replace("lobe","(lobe|lung)") + ")"))
        self.nodule_locs_re = nodule_locs_re
        
        # create regex to identify a potential location (for identifying right/left lung)
        self.all_locations = self.locations
        self.all_locations += ["left lung base", "right lung base", "left middle lung", "left mid lung",
                               "left lung apex", "right lung apex", "right lobe", "left lobe", "left lung",
                              "right lung", "left fissure", "right major fissure", "right minor fissure",
                              "rul", "rml", "rll", "lul", "lll", "lingula"]
        
        # regex to extract size
        self.nodule_size = re.compile('(([\d\.\-]+)((\s|\-){0,1}((m|c)m|millimet(er|re)|centimet(re|er))))')
        # regex to format/standardize the size of the nodule (either mm or cm for the units)
        self.nodule_size_formatter = re.compile("(?P<dash>[\-\s]{0,1})(?P<unit>((m|c)m|millimet(er|re)|centimet(re|er)))")
        # for extracting size out of a compound number (i.e. 7-8 mm)
        # we'll take the highest of the two numbers given a range
        compound_num_str = '(?P<start>\s|^)(?P<firstnum>\d+\.*\d*)(?P<middle>\sx\s)(?P<secondnum>\d\.*\d*)(?P<end>\s|$)'
        self.compound_num_re = re.compile(compound_num_str)
        
    # extract information about each nodule
    # runs on all phrases found for 'nodule' or equivalent
    def extract_nodules(self, phrases):
        nodules = []
        for phrase in phrases:
            size, numeric_size = self.extract_size(phrase['nodule_phrase'])
            location = self.extract_location(phrase['nodule_phrase'])
            lung = self.extract_lung(phrase['nodule_phrase'])
            phrase.update({'nodule_size': size, 'nodule_location': location, 'nodule_size_numeric': numeric_size,
                          'nodule_lung': lung, 'is_nodule': 1})
            nodules.append(phrase)
        # grabs the information about the largest nodule found
        max_nodule = self.get_max_nodule(nodules)
        max_nodule['max_nodule_size_numeric'] = self.get_numeric_size(max_nodule['max_nodule_size'])
        output = [{**x, **max_nodule} for x in nodules]                                                                              
        return output
        
    # extracts the actual location of the nodule
    def extract_location(self, phrase):
        for idx, nodule_loc_re in enumerate(self.nodule_locs_re):
            if nodule_loc_re.search(phrase):
                loc = self.locations[idx]
                if self.locations_dict.get(loc):
                    return self.locations_dict[loc]
                else:
                    return loc
        return ""
    
    # extract the lung location (left or right)
    def extract_lung(self, phrase):
        for location in self.all_locations:
            if location in phrase:
                if "left" in location or location[0] == 'l':
                    return "left"
                elif "right" in location or location[0] == 'r':
                    return "right"
        return ""
    
    # extracts the size of the nodule
    def extract_size(self, phrase):
        phrase = self.compound_to_simple_size(phrase)
        size_match = self.nodule_size.search(phrase)
        if size_match:
            size = self.nodule_size_formatter.sub(r" \g<unit>", size_match[0].strip())
            if size.startswith("-"):
                size = self.nodule_size_formatter.sub(r" \g<unit>", size[1:])
            elif re.search("\d\-\d", size):
                size = "".join(size.split("-")[1:])
            numeric_size = self.get_numeric_size(size)
            return size, numeric_size
        return "", 0
    
    # converts a compound size (i.e. 7-8 mm) to a simple size (i.e. 8 mm)
    def compound_to_simple_size(self, phrase):
        compound_search = self.compound_num_re.search(phrase)
        if compound_search:
            nums = []
            for group in compound_search.groups():
                if re.search('\d', group):
                    nums.append({'group': group, 'size': float(group)})
            largest = sorted(nums, key=itemgetter('size'), reverse=True)[0]
            return self.compound_num_re.sub(r"\g<start>{}\g<end>".format(largest['group']), phrase)
        else:
            return phrase
        
    # return the numeric size in millimeters
    def get_numeric_size(self, size):
        if size == '':
            return 0.0
        unit_size_dict = {'mm': 1, 'cm': 10, 'centimeter': 10, 'millimeter': 1,
                         'centimetre': 10, 'millimetre': 1}
        unit = unit_size_dict[size.split(" ")[1]]
        number = size.split(" ")[0]
        if "-" in number and not number.startswith("-"):
            number = number.split("-")[0]
        try:
            return float(number) * unit
        except:
            return 0.0
    
    # finds the largest nodule given a list of nodules
    def get_max_nodule(self, nodules):
        max_nodule = {'max_nodule_size': '', 'max_nodule_location': '', 'max_nodule_lung': '',
                         'max_nodule_size_numeric': 0}
        for nodule in nodules:
            if nodule['nodule_size_numeric'] > max_nodule['max_nodule_size_numeric']:
                max_nodule['max_nodule_size'] = nodule['nodule_size']
                max_nodule['max_nodule_size_numeric'] = nodule['nodule_size_numeric']
                max_nodule['max_nodule_location'] = nodule['nodule_location']
                max_nodule['max_nodule_lung'] = nodule['nodule_lung']
        return max_nodule