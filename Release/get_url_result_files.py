import requests
import json
####   将模型加载之后，通过url进行读取和使用

def run_url():

	url = "http://127.0.0.1:7777/Run_CLASS"
	payload_js = {
					"police_reason":"我在吃饭"
				  }
	headers = {'Content-Type': "application/json", }
	response = requests.request("POST", url, data=json.dumps(payload_js), headers=headers)
	print(json.loads(response.text))

if __name__ == "__main__":
	run_url()