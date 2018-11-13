import  requests

r = requests.get('https://www.douban.com/')
r.status_code

print (r.status_code)
print (r.text)