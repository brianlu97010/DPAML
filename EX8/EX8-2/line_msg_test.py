import requests

def lineNotifyMessage(token, msg):
    headers = { "Authorization": "Bearer " + token,"Content-Type" : "application/x-www-form-urlencoded" }
    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    return r.status_code

# 傳送訊息字串
message = "媽的死胖子 又摔倒了！"

# 修改成你的Token字串
token = '8YVSCjMlPlxuLJQtikTqmRXlEi0AuiKhqyyRW6P5yyA'
lineNotifyMessage(token, message)
