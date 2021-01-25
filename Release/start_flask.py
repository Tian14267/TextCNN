# -*- coding: utf-8 -*-
from flask import Flask
from flask import request
import json
import textcnn_pb
import os
path_now = os.getcwd()
model_path = os.path.join(path_now , "model_and_data/textcnn_twoClass.pb")
textclass = textcnn_pb.Textcnn_pred(model_path)
###  通过 flask 预加载模型，方便后面通过url进行数据加载和识别

app = Flask(__name__)
@app.route('/Run_CLASS',methods=['POST', 'GET'])
def run_test():
	content = request.get_data()
	print(json.loads(content)["police_reason"])
	content = json.loads(content)["police_reason"]
	index_result = textclass.text_pre(content)
	result = {
		'data': index_result
	}
	return result


if __name__ == "__main__":
	app.run(host='127.0.0.1', debug=True, port=7777, use_reloader=True, threaded=True)