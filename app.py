from flask import Flask
from flask import render_template, request, jsonify
import example
import json
import pandas as pd
from preproceesing_code import  *


app = Flask(__name__)

@app.route('/')
def get_comments():
    df = example.main()
    print(df.shape)
    #df.to_csv("comment.csv")
    print("praveen")
    preproceesing_fun(df)
    print("kumar")
    return render_template("index.html")


def main():
    app.run(port=3001, debug=True)


if __name__ == '__main__':
    main()