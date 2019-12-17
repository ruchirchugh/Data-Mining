import pandas as pd
import os, pickle, ast, operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def get_results(query):
    print("im get results")
    global pr_prob, po_prob
    initialize()
    if os.path.isfile("classifierPicklePrior.pkl"):
        pr_prob = pickle.load(open('classifierPicklePrior.pkl', 'rb'))
        po_prob = pickle.load(open('classifierPicklePost.pkl', 'rb'))
    else:
        (pr_prob, po_prob) = build_and_save()
    return eval_result(query)


def build_and_save():
    print("im build and save")
    row_count = 0
    t_c = 0
    po_prob = {}
    count_m = {}
    ge_count = {}
    for row in meta_data.itertuples():
        keywords = []
        genres = []
        for col in meta_cols.keys():
            c_val = getattr(row, col)
            parameters = meta_cols[col]
            # Paramter is None for tagline and overview columns, so appending data in keywords[]
            if parameters is None:
                keywords.append(c_val if isinstance(c_val, str) else "")
            # Else it is genres as it has a parameter "Name". So append in genres[]
            else:
                c_val = ast.literal_eval(c_val if isinstance(c_val, str) else '[]')
                for col_value in c_val:
                    for param in parameters:
                        genres.append(col_value[param])

        tokens = preprocessing(' '.join(keywords))
        for genre in genres:
            if genre in ge_count:
                ge_count[genre] += 1
            else:
                ge_count[genre] = 1
            for token in tokens:
                t_c += 1
                if (genre, token) in count_m:
                    count_m[(genre, token)] += 1
                else:
                    count_m[(genre, token)] = 1

        row_count += 1
        # Uncomment below lines for reading specific number of rows from excel instead of the whole
        # if (row_count == 2):
        #     print(ge_count)
        #     break
    for (genre, token) in count_m:
        po_prob[(genre, token)] = count_m[(genre, token)] / t_c

    pr_prob = {x: ge_count[x]/row_count for x in ge_count}
    save(pr_prob, po_prob)
    return (pr_prob, po_prob)
def eval_result(query):
    print("im eval")
    preprocss = preprocessing(query)
    genre_score = {}
    percentage = []
    sum = 0
    wrong_genre = ['Carousel Productions', 'Vision View Entertainment', 'Telescene Film Group Productions' , 'Aniplex' , 'GoHands' , 'BROSTA TV' , 'Mardock Scramble Production Committee' , 'Sentai Filmworks' , 'Odyssey Media' , 'Pulser Productions' , 'Rogue State' , 'The Cartel']
    for genre in pr_prob.keys():
        if (genre not in wrong_genre):
                score = pr_prob[genre]
                #print("For genre: ", genre, ", prior score: ", score)
                for token in preprocss:
                    if (genre, token) in po_prob.keys():
                        score = score * po_prob[(genre, token)]
                        #print("token: ", token, ", score: ", score)
                genre_score[genre] = score
    sorted_score_map = sorted(genre_score.items(), key=operator.itemgetter(1), reverse=True)
    for i in sorted_score_map[0:5]:
        sum+=i[1]
    for i in sorted_score_map[0:5]:
        percentage.append((i[1]/sum)*100)
    return sorted_score_map[0:5],percentage

def preprocessing(data_string):
    #print("im pre processing")
    # for noise in noise_list:
    #     data_string = data_string.replace(noise, "")
    tokens = tokenizer.tokenize(data_string)
    processed_data = []
    for t in tokens:
        if t not in stopword:
            processed_data.append(stemmer.stem(t).lower())
    return processed_data

def save(pr_prob, po_prob):
    print("im save")
    pickle.dump(pr_prob, open('classifierPicklePrior.pkl', 'wb+'))
    pickle.dump(po_prob, open('classifierPicklePost.pkl', 'wb+'))

def initialize():
    print("I,m Initialized")
    global meta_data, meta_cols, tokenizer, stopword, stemmer
    #data_folder = 'data/'
    meta_cols = {"genres":['name'], "overview":None, "tagline":None}
    meta_data = pd.read_csv('movies_metadata.csv', usecols=meta_cols.keys())

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
