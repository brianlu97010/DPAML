import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def waring_message():


    content = MIMEMultipart()  #建立MIMEMultipart物件
    content["subject"] = "A mail from MLVD"  #郵件標題
    content["from"] = "b092040010@g-mail.nsysu.edu.tw"  #寄件者
    content["to"] = "socool901107@gmail.com" #收件者
    content.attach(MIMEText("fat guy fall !!!"))  #郵件內容


    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        try:
            print("warning!!")  
            smtp.ehlo()  # 驗證SMTP伺服器
            smtp.starttls()  # 建立加密傳輸
            smtp.login("b092040010@g-mail.nsysu.edu.tw", "fsqi trsl ibwb mygz")  # 登入寄件者gmail
            smtp.send_message(content)  # 寄送郵件
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)
    