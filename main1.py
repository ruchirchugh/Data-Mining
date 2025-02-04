# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:58:06 2019

@author: Darshak
"""

import flask
from flask import Flask, render_template, request, jsonify
application = Flask(__name__)
application.debug = True
@application.route('/')
def hello_world():
    print("Hello")
    return render_template('index.html')

@application.route('/search/', methods=['GET', 'POST'])
def search():
    search_query = request.args.get('query','')
    print(search_query)
    import Take_2
    res, trimmed_query=Take_2.get_results(search_query)
    print(res[0:5])
    print(type(res))
    print("highlight: " ,trimmed_query)
    #data = {'results': res}
    #data = jsonify(data)
    return render_template('result.html',result=res[0:9], highlight = trimmed_query)

if __name__ == '__main__':
#    flask run
    #application.run(host='0.0.0.0',port=int()use_reloader=False)
   application.run(use_reloader=False)