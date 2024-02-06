# 한국어 -> 중국어 간체 -> 한국어
import urllib.request
from ast import literal_eval

client_id = "YOUR_ID" 
client_secret = "YOUR_KEY"

def ko_to_cn(text:str):
    encText = urllib.parse.quote(text)
    data = "source=ko&target=zh-CN&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        translated = literal_eval(response_body.decode('utf-8'))['message']['result']['translatedText']
    else:
        print("Error Code:" + rescode)
    
    return translated

def cn_to_ko(text:str):
    encText = urllib.parse.quote(text)
    data = "source=zh-CN&target=ko&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        translated = literal_eval(response_body.decode('utf-8'))['message']['result']['translatedText']
    else:
        print("Error Code:" + rescode)
    
    return translated

def papago_rtt(text:str):
    translated = ko_to_cn(text)
    return cn_to_ko(translated)