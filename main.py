from flask import Flask, render_template, request
#from datetime import datetime
#import plotClassification
import query_classifier
import tfidf
application = Flask(__name__)
application.debug = True
@application.route('/')
def hello_world():
    print("Hello")
    return render_template('index.html')

@application.route('/search/', methods=['GET', 'POST'])
def search():
    search_query = request.args.get('query','')
    # start time
    #start_time = datetime.now().timestamp()
    res, trimmed_query=tfidf.get_results(search_query)
    # end time
    #total_time = (datetime.now().timestamp() - start_time)
    #print(total_time)
    return render_template('result.html',result=res[15:30], highlight = trimmed_query)

@application.route('/classify/', methods=['GET', 'POST'])
def classify():
    search_query1 = request.args.get('query1','')
    results, percentages = query_classifier.get_results(search_query1)
    return render_template('result1.html', result=results , percentage = percentages)

@application.route('/caption/', methods=['GET', 'POST'])
def caption():
    search_query2 = request.args.get('query2','')
    results, highlights = tfidf.image_search(search_query2)
    return render_template('result2.html', result=results , highlight=highlights)

# if __name__ == '__main__':
#    flask run
    #application.run(host='0.0.0.0',port=int()use_reloader=False)
#   application.run(use_reloader=False)